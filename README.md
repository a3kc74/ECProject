# Time-Windowed Traveling Salesman Problem (TW-TSP) Optimization Framework

Một framework Python hoàn chỉnh để triển khai, so sánh và phân tích các thuật toán tối ưu hóa (metaheuristics) cho bài toán Time-Windowed TSP.

## 📋 Mô tả

Framework này cung cấp:
- **Triển khai bài toán TW-TSP** với ràng buộc cửa sổ thời gian
- **Thuật toán tối ưu hóa**: Genetic Algorithm (GA), Simulated Annealing (SA)
- **Framework Benchmark** để so sánh hiệu suất
- **Trực quan hóa** kết quả và lộ trình
- **Kiến trúc module** dễ mở rộng

## 🗂️ Cấu trúc Project

```
ECProject/
├── problem/
│   ├── __init__.py
│   └── tw_tsp.py              # Định nghĩa bài toán TW-TSP
├── algorithms/
│   ├── __init__.py
│   ├── base_algorithm.py      # Base class cho thuật toán
│   ├── genetic_algorithm.py   # Thuật toán di truyền
│   └── simulated_annealing.py # Thuật toán luyện kim
├── utils/
│   ├── __init__.py
│   └── visualizer.py          # Công cụ trực quan hóa
├── data/                      # Thư mục chứa dữ liệu problem
├── results/                   # Thư mục lưu kết quả
├── plots/                     # Thư mục lưu biểu đồ
├── benchmark.py               # Framework benchmark
├── main.py                    # Script chạy chính
├── requirements.txt           # Dependencies
└── README.md                  # File này
```

## 🚀 Cài đặt

### 1. Clone hoặc tải project

```bash
cd d:\Projects\ECProject
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 📊 Sử dụng

### Chạy thử nghiệm cơ bản

```bash
python main.py
```

Script sẽ:
1. Tạo hoặc tải problem instances
2. Chạy các thuật toán GA và SA
3. Thu thập và phân tích kết quả thống kê
4. Tạo các biểu đồ trực quan hóa

### Kết quả

Sau khi chạy, bạn sẽ có:

- **results/raw_results.csv**: Kết quả chi tiết của mỗi lần chạy
- **results/statistics.csv**: Thống kê tổng hợp (best, mean, std)
- **plots/convergence_comparison.png**: Đồ thị hội tụ
- **plots/benchmark_boxplot.png**: So sánh hiệu suất
- **plots/best_route_GA.png**: Lộ trình tốt nhất của GA
- **plots/best_route_SA.png**: Lộ trình tốt nhất của SA

## 🔧 Tùy chỉnh

### Thêm thuật toán mới

1. Tạo file mới trong `algorithms/`:

```python
from algorithms.base_algorithm import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def solve(self):
        # Triển khai thuật toán của bạn
        return best_solution, best_fitness, fitness_history
```

2. Import và thêm vào `main.py`:

```python
from algorithms.my_algorithm import MyAlgorithm

algorithms = {
    'GA': GeneticAlgorithm,
    'SA': SimulatedAnnealing,
    'MY': MyAlgorithm  # Thêm thuật toán mới
}
```

### Cấu hình thuật toán

Chỉnh sửa trong `main.py`:

```python
ga_config = {
    'population_size': 100,      # Kích thước quần thể
    'num_generations': 500,      # Số thế hệ
    'mutation_rate': 0.15,       # Tỷ lệ đột biến
    'crossover_rate': 0.85,      # Tỷ lệ lai ghép
    'tournament_size': 5,        # Kích thước tournament
    'elitism_count': 2           # Số cá thể ưu tú
}
```

### Sử dụng dữ liệu thật

1. Đặt file dữ liệu (định dạng Solomon) vào thư mục `data/`
2. Cập nhật `main.py`:

```python
problem_files = [
    'data/c101.txt',
    'data/c102.txt',
    'data/r101.txt'
]
```

## 📈 Bài toán TW-TSP

### Mô tả

Time-Windowed TSP là bài toán tìm lộ trình ngắn nhất đi qua tất cả khách hàng, với ràng buộc:

- Mỗi khách hàng có **cửa sổ thời gian** [ready_time, due_time]
- Đến **sớm** → phải chờ đợi
- Đến **trễ** → bị phạt nặng

### Hàm mục tiêu

```
Fitness = Total_Distance + Penalty * Σ(time_window_violations)
```

## 🧬 Thuật toán đã triển khai

### 1. Genetic Algorithm (GA)
- **Selection**: Tournament Selection
- **Crossover**: Ordered Crossover (OX)
- **Mutation**: Swap Mutation
- **Elitism**: Bảo toàn cá thể tốt nhất

### 2. Simulated Annealing (SA)
- **Neighbor**: 2-opt swap
- **Acceptance**: Metropolis criterion
- **Cooling**: Exponential schedule

## 📚 Dependencies

- `numpy>=1.24.0`: Tính toán số học
- `pandas>=2.0.0`: Xử lý dữ liệu
- `matplotlib>=3.7.0`: Trực quan hóa
- `scipy>=1.10.0`: Các hàm khoa học

## 🎯 Mục tiêu học tập

Framework này được thiết kế để:
1. Hiểu sâu về **Evolutionary Computation**
2. Thực hành **triển khai thuật toán metaheuristic**
3. Học cách **so sánh và đánh giá** thuật toán
4. Phát triển kỹ năng **lập trình Python** chuyên nghiệp
5. Làm quen với **Operations Research** thực tế

## 📝 Lưu ý

- Code sử dụng **Python 3.10+** với type hints
- Tất cả hàm đều có **docstring** chi tiết
- Framework được thiết kế **module** và dễ mở rộng
- Kết quả có thể **tái tạo** với random seed

## 🤝 Đóng góp

Để mở rộng framework:
1. Thêm thuật toán mới trong `algorithms/`
2. Thêm phương pháp trực quan hóa trong `utils/visualizer.py`
3. Thêm metrics đánh giá trong `benchmark.py`

## 📄 License

Project này được tạo cho mục đích học tập.

---

**Chúc bạn thành công với bài tập lớn! 🎓**
