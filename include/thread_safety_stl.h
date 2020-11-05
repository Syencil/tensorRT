// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/8/19

#ifndef TENSORRT_THREAD_SAFETY_STL_H
#define TENSORRT_THREAD_SAFETY_STL_H

#include <queue>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <future>

// 参考《CPP-Concurrency-In-Action-2ed-2019》
// front() 和 pop() 在多线程中会出现恶性竞争，导致线程无法获取想要的结果，所以此处将二者合并。try_pop不阻塞，获取成功为true否则false。pop_and_wait，阻塞直到获取元素。
// 例如: thread1和thread2都在获取数据，由于锁的粒度太小只能锁住数据而不是这两个操作。很有可能两个线程本来想要各读取一个变量，变成了两个线程读取了同一个变量
namespace tss{
    template<typename _Tp, typename _Sequence = std::deque<_Tp> >
    class thread_safety_queue{
    private:
        mutable std::mutex _mutex;
        std::condition_variable _con_empty;
        std::condition_variable _cv_full;
        const u_int32_t _max_size;

    protected:
        _Sequence c;

    public:
        thread_safety_queue() : _max_size(UINT32_MAX){
        }

        explicit thread_safety_queue(size_t size) : _max_size(size){
        }

        thread_safety_queue(thread_safety_queue const& other){
            std::lock_guard<std::mutex> lk(other._mutex);
            c = other.c;
        }

        size_t size() const{
            std::lock_guard<std::mutex> lk(_mutex);
            return c.size();
        }

        bool empty() const{
            std::lock_guard<std::mutex> lk(_mutex);
            return c.empty();
        }

        bool push(const _Tp& _Args){
            std::lock_guard<std::mutex> lk(_mutex);
            if(c.size() >= _max_size){
                return false;
            }
            c.push_back(_Args);
            _con_empty.notify_one();
            return true;
        }

        bool push(_Tp&& _Args){
            std::lock_guard<std::mutex> lk(_mutex);
            if(c.size()>=_max_size){
                return false;
            }
            c.push_back(std::move(_Args));
            _con_empty.notify_one();
            return true;
        }

        template <typename... _Args>
        bool emplace(_Args&&... __args){
            std::lock_guard<std::mutex> lk(_mutex);
            if(c.size()>=_max_size){
                return false;
            }
            c.emplace_back(std::forward<_Args>(__args)...);
            _con_empty.notify_one();
            return true;
        }

        void wait_and_pop(_Tp& _Args){
            std::unique_lock<std::mutex> lk(_mutex);
            _con_empty.wait(lk, [this](){return !c.empty();});
            _Args = c.front();
            c.pop_front();
        }

        std::shared_ptr<_Tp> wait_and_pop(){
            std::unique_lock<std::mutex> lk(_mutex);
            _con_empty.wait(lk, [this](){return !c.empty();});
            std::shared_ptr<_Tp> ptr(std::make_shared<_Tp>(c.front()));
            c.pop_front();
            return ptr;
        }

        bool try_pop(_Tp& _Args){
            std::lock_guard<std::mutex> lk(_mutex);
            if (c.empty()){
                return false;
            }
            _Args = c.front();
            c.pop_front();
            return true;
        }

        std::shared_ptr<_Tp> try_pop(){
            std::lock_guard<std::mutex> lk(_mutex);
            if (c.empty()){
                return std::shared_ptr<_Tp>();
            }
            std::shared_ptr<_Tp> ptr(std::make_shared<_Tp>(c.front()));
            c.pop_front();
            return ptr;
        }
    };
}

namespace tss{

    class thread_pool{
    private:
        std::atomic_bool _flag_done; // 不一定需要atomic_bool
        std::vector<std::thread> _threads;
        thread_safety_queue<std::function<void()>> _work_queue; // 任务队列 我们采用thread的方式所以没有返回值

    private:
        inline void run(){
            while(!_flag_done){
                std::function<void()> task;
                if(_work_queue.try_pop(task)){
                    task();
                }else{
                    std::this_thread::yield(); // 释放当前时间片
                }
            }
        }

    public:
         thread_pool() : _flag_done(false){
            unsigned int thread_num = std::thread::hardware_concurrency();
            if (0==thread_num){
                thread_num = 8;
                printf("Can not access hardware concurrency num !\n");
            }
            try{
                for (int i = 0; i<thread_num; ++i){
                    _threads.emplace_back(std::thread(&thread_pool::run, this));
                }
            }catch (...){
                _flag_done = true;
                throw;
            }
            printf("Thread Pool Created! Total num of threads is  %d\n", thread_num);
        }

        explicit thread_pool(unsigned int thread_num) : _flag_done(false){
            thread_num = std::min(std::thread::hardware_concurrency(), thread_num);
            try{
                for (int i = 0; i<thread_num; ++i){
                    _threads.emplace_back(std::thread(&thread_pool::run, this));
                }
            }catch (...){
                _flag_done = true;
                throw;
            }
            printf("Thread Pool Created! Total num of threads is  %d\n", thread_num);
        }

        template <typename _FunctionType, typename... _Args>
        std::future<typename std::result_of<_FunctionType(_Args...)>::type> submit(_FunctionType&& function, _Args&&... args){
            using return_type = typename std::result_of<_FunctionType(_Args...)>::type;
            auto task = std::make_shared<std::packaged_task<return_type ()>>(
                        std::bind(std::forward<_FunctionType>(function), std::forward<_Args>(args)...)
                    );
            std::future<return_type > res = task->get_future();
            _work_queue.emplace([task](){(*task)(); });
            return res;
        } // 使用typename表示type是类型不是变量

        ~thread_pool(){
            _flag_done = true;
            for(auto & t : _threads){
                if(t.joinable()){
                    t.join();
                }
            }
            printf("Thread pool is closed\n");
        }

        void shutdown() noexcept {
            printf("Trying to shut down pool...\n");
            _flag_done = true;
        }

        thread_pool(const thread_pool &other) = delete;
        thread_pool(thread_pool &other) = delete;
        thread_pool &operator=(const thread_pool&) = delete;
    };
}




#endif //TENSORRT_THREAD_SAFETY_STL_H
