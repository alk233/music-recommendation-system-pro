import threading
import time

from config import DEFAULT_RECOMMEND


_LOCK = threading.Lock()
_JOB = None
_RESULTS = {}


def _signature(username, user_id, user_history, topk):
    hist = tuple(sorted(set(int(x) for x in (user_history or []))))
    return (str(username or ""), int(user_id or -1), hist, int(topk))


def cancel_idle_prefetch():
    global _JOB
    with _LOCK:
        job = _JOB
        if job is not None:
            job["cancel"].set()
            _JOB = None


def get_prefetch_result(task_key, username, user_id, user_history, topk=DEFAULT_RECOMMEND):
    sig = _signature(username, user_id, user_history, topk)
    with _LOCK:
        return _RESULTS.get((sig, task_key))


def _run_task(task_key, fn, sig, cancel_event):
    if cancel_event.is_set():
        return
    try:
        result = fn()
    except Exception:
        result = None
    if cancel_event.is_set():
        return
    with _LOCK:
        _RESULTS[(sig, task_key)] = result


def _worker(sig, user_id, user_history, cancel_event):
    # 延迟一点点，给用户即时点击让路
    time.sleep(0.4)
    if cancel_event.is_set():
        return

    from src.recommend_utils import (
        usercf_topn,
        itemcf_topn,
        svd_topn,
        ncf_recommend,
        content_based_recommend,
        hybrid_recommend,
    )

    tasks = [
        ("cf_usercf", lambda: usercf_topn(user_id, DEFAULT_RECOMMEND, user_history=user_history)),
        ("cf_itemcf", lambda: itemcf_topn(user_id, DEFAULT_RECOMMEND, user_history=user_history)),
        ("cf_svd", lambda: svd_topn(user_id, DEFAULT_RECOMMEND, user_history=user_history)),
        ("ncf", lambda: ncf_recommend(user_id, DEFAULT_RECOMMEND, user_history)),
        ("content", lambda: content_based_recommend(user_history, DEFAULT_RECOMMEND)),
        ("hybrid", lambda: hybrid_recommend(user_id, user_history, DEFAULT_RECOMMEND, None)),
    ]
    for task_key, fn in tasks:
        if cancel_event.is_set():
            return
        _run_task(task_key, fn, sig, cancel_event)


def start_idle_prefetch(username, user_id, user_history, topk=DEFAULT_RECOMMEND):
    global _JOB
    if user_id is None:
        return
    if int(topk) != int(DEFAULT_RECOMMEND):
        return
    if not user_history:
        return

    sig = _signature(username, user_id, user_history, topk)
    with _LOCK:
        if _JOB is not None:
            same = _JOB.get("sig") == sig and _JOB["thread"].is_alive()
            if same:
                return
            _JOB["cancel"].set()
            _JOB = None

        cancel_event = threading.Event()
        t = threading.Thread(
            target=_worker,
            args=(sig, int(user_id), list(user_history), cancel_event),
            daemon=True,
        )
        _JOB = {"sig": sig, "cancel": cancel_event, "thread": t}
        t.start()

