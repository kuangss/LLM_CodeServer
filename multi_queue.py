import threading
import queue
from typing import Generator, Callable, Any, List, Tuple
import time
from concurrent.futures import ThreadPoolExecutor

class ParallelGenerators:
    """
    并行执行多个生成器的类
    """
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers
        self._result_queue = queue.Queue()
        self._done_queue = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    
    def add_generator(self, 
                     generator_func: Callable, 
                     *args, 
                     **kwargs) -> None:
        """添加一个生成器任务"""
        worker_id = len(self._workers)
        future = self._thread_pool.submit(
            self._run_generator,
            generator_func, args, kwargs, worker_id
        )
        # 创建一个轻量级的线程兼容对象
        class WorkerWrapper:
            def __init__(self, future):
                self.future = future
            
            def is_alive(self):
                return not self.future.done()
            
            def join(self, timeout=None):
                if not self.future.done():
                    try:
                        self.future.result(timeout=timeout)
                    except Exception:
                        pass  # 异常已在 _run_generator 中处理
        
        self._workers.append(WorkerWrapper(future))
    
    def _run_generator(self, 
                      generator_func: Callable, 
                      args: Tuple, 
                      kwargs: dict, 
                      worker_id: int):
        """在单独线程中运行生成器"""
        try:
            for item in generator_func(*args, **kwargs):
                self._result_queue.put(item)
        except Exception as e:
            self._result_queue.put(("ERROR", str(e)))
        finally:
            self._done_queue.put(worker_id)
    
    def start(self) -> Generator[Any, None, None]:
        """启动所有生成器并返回统一的结果生成器"""
        # 返回生成器
        completed = set()
        
        while True:
            # 首先尝试获取所有可用的结果
            items_yielded = False
            while True:
                try:
                    item = self._result_queue.get_nowait()
                    # 更精确的错误检测
                    if isinstance(item, tuple) and len(item) == 2 and item[0] == "ERROR":
                        raise RuntimeError(f"生成器错误: {item[1]}")
                    yield item
                    items_yielded = True
                except queue.Empty:
                    break
            
            # 检查完成信号
            while True:
                try:
                    done_id = self._done_queue.get_nowait()
                    completed.add(done_id)
                except queue.Empty:
                    break
            
            # 如果所有线程都完成了，处理剩余结果并退出
            if len(completed) >= len(self._workers):
                # 获取所有剩余结果
                while not self._result_queue.empty():
                    item = self._result_queue.get_nowait()
                    if isinstance(item, tuple) and item[0] == "ERROR":
                        raise RuntimeError(f"生成器错误: {item[1]}")
                    yield item
                break
            
            # 如果没有产出任何项目且线程仍在运行，短暂等待
            if not items_yielded:
                if all(not t.is_alive() for t in self._workers):
                    # 所有线程都已结束，获取剩余结果
                    while not self._result_queue.empty():
                        item = self._result_queue.get_nowait()
                        if isinstance(item, tuple) and item[0] == "ERROR":
                            raise RuntimeError(f"生成器错误: {item[1]}")
                        yield item
                    break
                # 短暂等待避免CPU占用过高
                time.sleep(0.01)
        
        # 等待所有任务完成
        for thread in self._workers:
            thread.join(timeout=1)
    
    def shutdown(self, wait: bool = True):
        """关闭线程池"""
        self._thread_pool.shutdown(wait=wait)
    
    def __enter__(self):
        """支持上下文管理器"""
        return self
    
    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """退出上下文时自动关闭线程池"""
        self.shutdown()
        return False  # 不抑制异常
    
    def __call__(self, *generator_configs) -> Generator[Any, None, None]:
        """快捷调用方式"""
        for config in generator_configs:
            if isinstance(config, tuple) and len(config) >= 2:
                self.add_generator(config[0], *config[1])
            else:
                self.add_generator(config)
        
        return self.start()

# 使用示例
def example_generator(name: str, count: int):
    for i in range(count):
        time.sleep(0.1)  # 减少延迟用于测试
        yield f"{name}: {i}"

# 依赖链任务示例
def stage1_processor(data: List[int]) -> Generator[str, None, None]:
    """第一阶段：数据处理"""
    for i, item in enumerate(data):
        time.sleep(0.05)
        yield f"Stage1-Processed-{i}: {item * 2}"

def stage2_processor(stage1_results: List[str]) -> Generator[str, None, None]:
    """第二阶段：基于第一阶段结果的处理"""
    for i, result in enumerate(stage1_results):
        time.sleep(0.05)
        # 提取数字并继续处理
        number = int(result.split(': ')[1])
        yield f"Stage2-Enhanced-{i}: {number + 100}"

def stage3_processor(stage2_results: List[str]) -> Generator[str, None, None]:
    """第三阶段：最终处理"""
    for i, result in enumerate(stage2_results):
        time.sleep(0.05)
        number = int(result.split(': ')[1])
        yield f"Final-Result-{i}: {number ** 0.5:.2f}"

def dependent_pipeline_example():
    """依赖管道示例：每个阶段依赖前一阶段的结果"""
    print("=== 依赖管道示例 ===")
    
    # 初始数据
    initial_data = [1, 2, 3, 4, 5]
    
    # 第一阶段
    print("阶段1：处理初始数据...")
    with ParallelGenerators(max_workers=3) as stage1:
        # 将数据分批并行处理
        batch_size = 2
        for i in range(0, len(initial_data), batch_size):
            batch = initial_data[i:i+batch_size]
            stage1.add_generator(stage1_processor, batch)
        
        stage1_results = list(stage1.start())
        print(f"阶段1完成，得到 {len(stage1_results)} 个结果:")
        for result in stage1_results:
            print(f"  {result}")
    
    # 第二阶段：依赖第一阶段结果
    print("\n阶段2：基于阶段1结果进行处理...")
    with ParallelGenerators(max_workers=2) as stage2:
        batch_size = 3
        for i in range(0, len(stage1_results), batch_size):
            batch = stage1_results[i:i+batch_size]
            stage2.add_generator(stage2_processor, batch)
        
        stage2_results = list(stage2.start())
        print(f"阶段2完成，得到 {len(stage2_results)} 个结果:")
        for result in stage2_results:
            print(f"  {result}")
    
    # 第三阶段：依赖第二阶段结果
    print("\n阶段3：最终处理...")
    with ParallelGenerators(max_workers=2) as stage3:
        batch_size = 2
        for i in range(0, len(stage2_results), batch_size):
            batch = stage2_results[i:i+batch_size]
            stage3.add_generator(stage3_processor, batch)
        
        final_results = list(stage3.start())
        print(f"阶段3完成，得到 {len(final_results)} 个最终结果:")
        for result in final_results:
            print(f"  {result}")
    
    print("\n=== 管道处理完成 ===")

def adaptive_dependent_example():
    """自适应依赖示例：根据前一阶段结果动态调整下一阶段"""
    print("\n=== 自适应依赖示例 ===")
    
    # 阶段1：数据收集
    def data_collector(source_id: int) -> Generator[str, None, None]:
        for i in range(3):
            time.sleep(0.02)
            yield f"Data-Source{source_id}-Item{i}: {source_id * 10 + i}"
    
    with ParallelGenerators(max_workers=3) as collector:
        for source_id in range(1, 4):
            collector.add_generator(data_collector, source_id)
        
        collected_data = list(collector.start())
        print(f"收集到 {len(collected_data)} 条数据:")
        for data in collected_data:
            print(f"  {data}")
    
    # 阶段2：根据数据量动态分配处理器
    def data_processor(data_batch: List[str]) -> Generator[str, None, None]:
        for i, data in enumerate(data_batch):
            time.sleep(0.03)
            # 提取数值进行处理
            value = int(data.split(': ')[1])
            processed_value = value * value
            yield f"Processed-{i}: {processed_value}"
    
    # 根据数据量动态调整批次大小
    batch_size = max(1, len(collected_data) // 2)
    print(f"\n根据数据量 {len(collected_data)}，设置批次大小为 {batch_size}")
    
    with ParallelGenerators(max_workers=2) as processor:
        for i in range(0, len(collected_data), batch_size):
            batch = collected_data[i:i+batch_size]
            processor.add_generator(data_processor, batch)
        
        processed_data = list(processor.start())
        print(f"处理完成，得到 {len(processed_data)} 个结果:")
        for result in processed_data:
            print(f"  {result}")
    
    # 阶段3：结果聚合
    def result_aggregator(results: List[str]) -> Generator[str, None, None]:
        total = sum(int(r.split(': ')[1]) for r in results)
        time.sleep(0.01)
        yield f"Aggregated-Total: {total}"
        yield f"Aggregated-Average: {total / len(results):.2f}"
        yield f"Aggregated-Count: {len(results)}"
    
    with ParallelGenerators(max_workers=1) as aggregator:
        aggregator.add_generator(result_aggregator, processed_data)
        
        final_stats = list(aggregator.start())
        print(f"\n最终统计结果:")
        for stat in final_stats:
            print(f"  {stat}")
    
    print("=== 自适应处理完成 ===")

def real_time_pipeline_example():
    """实时流水线示例：第一循环产生结果后立即启动第二循环"""
    print("=== 实时流水线示例 ===")
    
    # 第一阶段：数据生产者
    def data_producer() -> Generator[str, None, None]:
        """生产数据批次"""
        for batch_id in range(1, 4):
            for i in range(3):
                time.sleep(0.05)
                yield f"Batch{batch_id}-Item{i}: {batch_id * 100 + i}"
    
    # 第二阶段：实时处理器
    def real_time_processor(item: str) -> Generator[str, None, None]:
        """处理单个数据项"""
        time.sleep(0.03)
        value = int(item.split(': ')[1])
        processed = value * 2
        yield f"Processed-{item}: {processed}"
    
    # 第三阶段：最终聚合器
    def final_aggregator(processed_item: str) -> Generator[str, None, None]:
        """最终聚合处理"""
        time.sleep(0.02)
        original = processed_item.split('Processed-')[1]
        value = int(original.split(': ')[1])
        yield f"Final-{value}: {value ** 0.5:.2f}"
    
    print("启动实时流水线处理...")
    
    # 使用嵌套的ParallelGenerators实现流水线
    final_results = []
    
    with ParallelGenerators(max_workers=2) as producers:
        # 启动生产者
        producers.add_generator(data_producer)
        
        # 对每个生产结果立即启动处理
        with ParallelGenerators(max_workers=3) as processors:
            with ParallelGenerators(max_workers=2) as aggregators:
                
                for produced_item in producers.start():
                    print(f"📦 生产: {produced_item}")
                    
                    # 立即为此项启动处理器
                    processors.add_generator(real_time_processor, produced_item)
                    
                    # 收集处理结果
                    processed_items = list(processors.start())
                    for processed_item in processed_items:
                        print(f"⚙️  处理: {processed_item}")
                        
                        # 立即为此项启动聚合器
                        aggregators.add_generator(final_aggregator, processed_item)
                        
                        # 收集最终结果
                        final_items = list(aggregators.start())
                        for final_item in final_items:
                            final_results.append(final_item)
                            print(f"✅ 聚合: {final_item}")
    
    print(f"\n流水线完成！最终处理了 {len(final_results)} 个项目")

def streaming_dependent_example():
    """流式依赖示例：基于前一个结果动态决定下一步"""
    print("\n=== 流式依赖示例 ===")
    
    # 动态决策处理器
    def dynamic_processor(data: str) -> Generator[str, None, None]:
        """根据输入动态决定处理策略"""
        value = int(data.split(': ')[1])
        
        if value < 150:
            # 小值：简单处理
            time.sleep(0.02)
            yield f"Simple-{data}: {value + 10}"
        elif value < 200:
            # 中值：复杂处理
            time.sleep(0.04)
            yield f"Complex-{data}: {value * 1.5}"
        else:
            # 大值：多重处理
            time.sleep(0.06)
            yield f"Multi-{data}: {value ** 2}"
    
    # 后续处理器
    def follow_up_processor(processed_data: str) -> Generator[str, None, None]:
        """根据前一步结果进行后续处理"""
        if "Simple" in processed_data:
            time.sleep(0.01)
            value = int(processed_data.split(': ')[1])
            yield f"FollowUp-Simple: {value * 3}"
        elif "Complex" in processed_data:
            time.sleep(0.02)
            value = int(processed_data.split(': ')[1])
            yield f"FollowUp-Complex: {value / 2}"
        else:  # Multi
            time.sleep(0.03)
            value = int(processed_data.split(': ')[1])
            yield f"FollowUp-Multi: {value ** 0.5}"
    
    print("启动流式依赖处理...")
    
    with ParallelGenerators(max_workers=3) as stage1:
        with ParallelGenerators(max_workers=2) as stage2:
            
            # 第一阶段：初始数据生成
            def initial_data_generator():
                for i in range(5):
                    time.sleep(0.03)
                    yield f"Data-{i}: {100 + i * 25}"
            
            stage1.add_generator(initial_data_generator)
            
            # 流式处理
            for data in stage1.start():
                print(f"📊 输入: {data}")
                
                # 立即启动动态处理器
                stage2.add_generator(dynamic_processor, data)
                
                # 获取动态处理结果
                for processed in stage2.start():
                    print(f"🔄 动态处理: {processed}")
                    
                    # 立即启动后续处理
                    stage2.add_generator(follow_up_processor, processed)
                    
                    for final in stage2.start():
                        print(f"🎯 最终结果: {final}")

def main3():
    # 使用上下文管理器自动管理线程池
    with ParallelGenerators(max_workers=4) as parallel:
        # 添加多个生成器
        parallel.add_generator(example_generator, "Worker1", 3)
        parallel.add_generator(example_generator, "Worker2", 5)
        parallel.add_generator(example_generator, "Worker3", 2)
        
        # 获取统一生成器并消费
        for result in parallel.start():
            print(result)
    
    print("\n--- 使用快捷方式 ---\n")
    
    # 快捷方式（手动管理线程池）
    parallel2 = ParallelGenerators(max_workers=2)
    try:
        gen = parallel2(
            (example_generator, ["A", 2]),
            (example_generator, ["B", 4]),
            (example_generator, ["C", 3])
        )
        
        for result in gen:
            print(result)
    finally:
        # 手动关闭线程池
        parallel2.shutdown()

def main():
    """主函数：运行所有示例"""
    print("=== 基础并行示例 ===")
    main3()
    
    # 运行依赖管道示例
    dependent_pipeline_example()
    
    # 运行自适应依赖示例
    adaptive_dependent_example()
    
    # 运行实时流水线示例
    real_time_pipeline_example()
    
    # 运行流式依赖示例
    streaming_dependent_example()

def demo_realtime_only():
    """仅演示实时流水线示例"""
    real_time_pipeline_example()

if __name__ == "__main__":
    # main()  # 运行所有示例
    demo_realtime_only()  # 仅运行实时流水线示例