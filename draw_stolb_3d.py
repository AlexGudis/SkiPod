import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Функция для построения 3D-столбчатого графика
def plot_3d_chart(data, threads_range, title, save_as):
    matrix_sizes = data.index.values
    threads = data.columns.values
    execution_times = data.values

    # Фильтрация данных по числу потоков
    thread_indices = np.where((threads >= threads_range[0]) & (threads <= threads_range[1]))[0]
    threads_filtered = threads[thread_indices]
    execution_times_filtered = execution_times[:, thread_indices]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    bar_width = 10.0 
    for i, matrix_size in enumerate(matrix_sizes):
        print(i)
        ax.bar(
            threads_filtered, 
            execution_times_filtered[i], 
            zs=matrix_size, 
            zdir='y', 
            alpha=0.9, 
            width=bar_width
            
        )

    ax.set_xlabel("Число потоков")
    ax.set_ylabel("Размер матрицы, N")
    ax.set_zlabel("Время работы, c")
    ax.set_title(title, fontsize=14)
    ax.set_zlim(0, np.max(execution_times_filtered) * 1.5)

    plt.tight_layout()
    plt.savefig(save_as)
    plt.show()

file_path = "./mpi_data.xlsx"
sheet_name = "Лист1"

data = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)

plot_3d_chart(data, threads_range=(1, 10), 
              title="Зависимость времени выполнения от числа потоков и размерности матрицы. Технология MPI. Потоки 1-10.", 
              save_as="graph_1_10.png")

plot_3d_chart(data, threads_range=(10, 160), 
              title="Зависимость времени выполнения от числа потоков и размерности матрицы. Технология MPI. Потоки 10-160", 
              save_as="graph_10_160.png")
