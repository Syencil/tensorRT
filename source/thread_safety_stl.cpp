// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/8/19

#include "thread_safety_stl.h"

template<typename _Tp, typename _Sequence>
thread_safety_queue<_Tp, _Sequence>::thread_safety_queue(thread_safety_queue const &other){
    std::lock_guard<std::mutex> lk(other._mutex);
    c = other.c;
}

template<typename _Tp, typename _Sequence>
bool thread_safety_queue<_Tp, _Sequence>::empty() const {
    std::lock_guard<std::mutex> lk(_mutex);
    return c.empty();
}

template<typename _Tp, typename _Sequence>
void thread_safety_queue<_Tp, _Sequence>::push(const _Tp &_Args) {
    std::lock_guard<std::mutex> lk(_mutex);
    c.push(_Args);
    _con_var.notify_one();
}

template<typename _Tp, typename _Sequence>
void thread_safety_queue<_Tp, _Sequence>::push(_Tp &&_Args) {
    std::lock_guard<std::mutex> lk(_mutex);
    c.push(_Args);
    _con_var.notify_one();
}

template<typename _Tp, typename _Sequence>
template<typename... _Args>
void thread_safety_queue<_Tp, _Sequence>::emplace(_Args &&... __args) {
    std::lock_guard<std::mutex> lk(_mutex);
    c.emplace_back(std::forward<_Args>(__args)...);
    _con_var.notify_one();
}

template<typename _Tp, typename _Sequence>
void thread_safety_queue<_Tp, _Sequence>::wait_and_pop(_Tp &_Args) {
    std::unique_lock<std::mutex> lk(_mutex);
    _con_var.wait(lk, [this](){return !c.empty();});
    _Args = c.front();
    c.pop();
}

template<typename _Tp, typename _Sequence>
std::shared_ptr<_Tp> thread_safety_queue<_Tp, _Sequence>::wait_and_pop() {
    std::unique_lock<std::mutex> lk(_mutex);
    _con_var.wait(lk, [this](){return !c.empty();});
    std::shared_ptr<_Tp> ptr(std::make_shared<_Tp>(c.front));
    c.pop();
    return ptr;
}

template<typename _Tp, typename _Sequence>
bool thread_safety_queue<_Tp, _Sequence>::try_pop(_Tp &_Args) {
    std::lock_guard<std::mutex> lk(_mutex);
    if (c.empty()){
        return false;
    }
    _Args = c.front();
    c.pop();
    return true;
}

template<typename _Tp, typename _Sequence>
std::shared_ptr<_Tp> thread_safety_queue<_Tp, _Sequence>::try_pop() {
    std::lock_guard<std::mutex> lk(_mutex);
    if (c.empty()){
        return std::shared_ptr<_Tp>();
    }
    std::shared_ptr<_Tp> ptr(std::make_shared<_Tp>(c.front()));
    c.pop();
    return ptr;
}
