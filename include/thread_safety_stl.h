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
        std::condition_variable _con_var;

    protected:
        _Sequence c;

    public:
        thread_safety_queue()= default;

        thread_safety_queue(thread_safety_queue const& other);

        bool empty() const;

        void push(const _Tp& _Args);

        void push(_Tp&& _Args);

        template <typename... _Args>
        void emplace(_Args&&... __args);

        void wait_and_pop(_Tp& _Args);

        std::shared_ptr<_Tp> wait_and_pop();

        bool try_pop(_Tp& _Args);

        std::shared_ptr<_Tp> try_pop();
    };
}

namespace tss{
    class thread_joiner{
    private:
        std::vector<std::thread> &_threads;
    public:
        explicit thread_joiner(std::vector<std::thread>& threads) : _threads(threads){

        };
        ~thread_joiner(){
            for(auto & t : _threads){
                if(t.joinable()){
                    t.join();
                }
            }
        }
    };
}

namespace tss{
    class thread_pool{
    private:
        bool _flag_done; // 不需要atomic
        std::vector<std::thread> _threads;
        tss::thread_joiner _joiner;
        thread_safety_queue<std::function<void()>> _work_queue; // 任务队列 我们采用thread的方式所以没有返回值

    private:
        void run();

    public:
        explicit thread_pool();

        explicit thread_pool(int thread_num);

        template <typename _FunctionType>
        std::future<typename std::result_of<_FunctionType()>::type> submit(_FunctionType function); // 使用typename表示type是类型不是变量

        ~thread_pool();

        thread_pool(const thread_pool &other) = delete;
        thread_pool(thread_pool &other) = delete;
        thread_pool &operator=(const thread_pool&) = delete;
    };
}




#endif //TENSORRT_THREAD_SAFETY_STL_H
