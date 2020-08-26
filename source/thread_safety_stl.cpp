// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/8/19

#include "thread_safety_stl.h"
namespace tss{
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
        c.push_back(_Args);
        _con_var.notify_one();
    }

    template<typename _Tp, typename _Sequence>
    void thread_safety_queue<_Tp, _Sequence>::push(_Tp &&_Args) {
        std::lock_guard<std::mutex> lk(_mutex);
        c.push_back(_Args);
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
        c.pop_front();
    }

    template<typename _Tp, typename _Sequence>
    std::shared_ptr<_Tp> thread_safety_queue<_Tp, _Sequence>::wait_and_pop() {
        std::unique_lock<std::mutex> lk(_mutex);
        _con_var.wait(lk, [this](){return !c.empty();});
        std::shared_ptr<_Tp> ptr(std::make_shared<_Tp>(c.front()));
        c.pop_front();
        return ptr;
    }

    template<typename _Tp, typename _Sequence>
    bool thread_safety_queue<_Tp, _Sequence>::try_pop(_Tp &_Args) {
        std::lock_guard<std::mutex> lk(_mutex);
        if (c.empty()){
            return false;
        }
        _Args = c.front();
        c.pop_front();
        return true;
    }

    template<typename _Tp, typename _Sequence>
    std::shared_ptr<_Tp> thread_safety_queue<_Tp, _Sequence>::try_pop() {
        std::lock_guard<std::mutex> lk(_mutex);
        if (c.empty()){
            return std::shared_ptr<_Tp>();
        }
        std::shared_ptr<_Tp> ptr(std::make_shared<_Tp>(c.front()));
        c.pop_front();
        return ptr;
    }
}

namespace tss{
    thread_pool::thread_pool() : _joiner(_threads), _flag_done(false){
        unsigned int thread_num = std::thread::hardware_concurrency();
        if (0==thread_num){
            thread_num = 8;
            printf("Can not access hardware concurrency num !\n");
        }
        try{
            for (int i = 0; i<thread_num; ++i){
                _threads.emplace_back(std::thread(&thread_pool::_work_queue, this));
            }
        }catch (...){
            _flag_done = true;
            throw;
        }
        printf("Thread num = %d\n", thread_num);
        run();
    }

    thread_pool::thread_pool(int thread_num) : _joiner(_threads), _flag_done(false){
        try{
            for (int i = 0; i<thread_num; ++i){
                _threads.emplace_back(std::thread(&thread_pool::_work_queue, this));
            }
        }catch (...){
            _flag_done = true;
            throw;
        }
        printf("Thread num = %d\n", thread_num);
        run();
    }

    template<typename _FunctionType>
    std::future<typename std::result_of<_FunctionType()>::type> thread_pool::submit(_FunctionType function) {
        typedef typename std::result_of<_FunctionType()>::type result_type;
        std::packaged_task<result_type()> task(std::move(function));
        std::future<result_type> res(task.get_future());
        _work_queue.emplace(std::move(task));
        return res;
    }

    void thread_pool::run() {
        while(!_flag_done){
            std::function<void()> task;
            _work_queue.wait_and_pop();
            if(_work_queue.try_pop(task)){
                task();
            }else{
                std::this_thread::yield(); // 释放当前时间片
            }
        }
    }

    thread_pool::~thread_pool() {
        _flag_done = true;
    }

}