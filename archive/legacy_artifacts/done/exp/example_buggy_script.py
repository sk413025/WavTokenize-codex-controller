#!/usr/bin/env python3
"""
簡單的示範腳本 - 包含一個隱藏的 bug

這個腳本計算一個列表的平均值，但在某些情況下會出錯。
"""

def calculate_average(numbers):
    """計算數字列表的平均值"""
    total = 0
    count = 0

    for num in numbers:
        total += num
        count += 1

    # Bug: 當列表為空時會除以零
    average = total / count
    return average


def process_data(data_list):
    """處理多組數據"""
    results = []

    for i, data in enumerate(data_list):
        print(f"Processing dataset {i}...")
        avg = calculate_average(data)
        results.append(avg)
        print(f"  Average: {avg}")

    return results


def main():
    """主函數"""
    # 測試數據
    datasets = [
        [1, 2, 3, 4, 5],           # 正常數據
        [10, 20, 30],              # 正常數據
        [],                        # Bug: 空列表會導致除零錯誤
        [100, 200, 300],           # 這個不會被執行到（因為前面會出錯）
    ]

    print("Starting data processing...")
    results = process_data(datasets)
    print(f"\nAll results: {results}")
    print("Done!")


if __name__ == "__main__":
    main()
