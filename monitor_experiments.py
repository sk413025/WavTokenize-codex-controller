#!/usr/bin/env python3
"""
實驗監控系統
監控所有運行中的Python進程和實驗進度
"""
import subprocess
import time
import os
import re
from datetime import datetime
from typing import List, Dict, Optional

class ExperimentMonitor:
    def __init__(self):
        """實驗監控器初始化"""
        self.monitor_interval = 30  # 監控間隔秒數
        self.log_dir = "/home/sbplab/ruizi/c_code/logs"
        self.terminal_ids = []  # 追蹤的終端ID
        
    def get_running_processes(self) -> List[Dict]:
        """獲取所有運行中的Python/實驗進程"""
        try:
            cmd = ["ps", "aux"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            processes = []
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['python', 'ttt2', 'wavtokenizer', 'discrete']):
                    if 'grep' not in line and 'ps aux' not in line:
                        parts = line.split()
                        if len(parts) >= 11:
                            process_info = {
                                'user': parts[0],
                                'pid': parts[1],
                                'cpu': parts[2],
                                'memory': parts[3],
                                'command': ' '.join(parts[10:])
                            }
                            processes.append(process_info)
            
            return processes
        except subprocess.CalledProcessError as e:
            print(f"❌ 獲取進程失敗: {e}")
            return []

    def get_latest_logs(self) -> Dict[str, str]:
        """獲取最新的實驗日誌檔案"""
        log_files = {}
        try:
            # 查找token/discrete相關日誌
            for pattern in ['*token*', '*discrete*', '*ttt2*']:
                cmd = ["find", self.log_dir, "-name", pattern, "-type", "f"]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                for log_file in result.stdout.strip().split('\n'):
                    if log_file and os.path.exists(log_file):
                        # 獲取檔案修改時間
                        mtime = os.path.getmtime(log_file)
                        log_files[log_file] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                        
        except subprocess.CalledProcessError as e:
            print(f"❌ 獲取日誌檔案失敗: {e}")
            
        return log_files

    def check_token_loss_status(self, log_file: str) -> Optional[Dict]:
        """檢查token loss實驗狀態"""
        try:
            # 檢查最後100行日誌
            cmd = ["tail", "-100", log_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            lines = result.stdout.split('\n')
            status = {
                'has_token_loss_errors': False,
                'latest_error': None,
                'current_epoch': None,
                'progress': None,
                'dimension_errors': []
            }
            
            for line in lines:
                # 檢查token loss錯誤
                if 'Token loss 計算失敗' in line:
                    status['has_token_loss_errors'] = True
                    # 提取維度錯誤信息
                    if 'size of tensor' in line:
                        match = re.search(r'size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\)', line)
                        if match:
                            dim_a, dim_b = match.groups()
                            status['dimension_errors'].append(f"{dim_a} vs {dim_b}")
                            status['latest_error'] = f"維度不匹配: {dim_a} vs {dim_b}"
                
                # 檢查訓練進度
                if 'Epoch' in line and '%' in line:
                    epoch_match = re.search(r'Epoch (\d+)', line)
                    progress_match = re.search(r'(\d+)%', line)
                    if epoch_match:
                        status['current_epoch'] = epoch_match.group(1)
                    if progress_match:
                        status['progress'] = f"{progress_match.group(1)}%"
                        
            return status
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 檢查日誌失敗 {log_file}: {e}")
            return None

    def check_terminal_output(self, terminal_id: str) -> Optional[str]:
        """檢查指定終端的輸出狀態"""
        # 這裡只能返回狀態描述，因為無法直接訪問VS Code的終端輸出
        return f"Terminal {terminal_id[:8]}... - 需要手動檢查VS Code終端"

    def generate_report(self) -> str:
        """生成監控報告"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    🔍 實驗監控報告                                                  ║
║                                    時間: {timestamp}                                    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

📊 運行中的進程:
"""
        
        processes = self.get_running_processes()
        if processes:
            for i, proc in enumerate(processes, 1):
                report += f"""
{i}. PID: {proc['pid']} | CPU: {proc['cpu']}% | Memory: {proc['memory']}% | User: {proc['user']}
   Command: {proc['command'][:100]}{'...' if len(proc['command']) > 100 else ''}
"""
        else:
            report += "\n   ❌ 未發現相關實驗進程\n"

        # 檢查日誌檔案
        log_files = self.get_latest_logs()
        report += f"\n📝 最新日誌檔案 ({len(log_files)} 個):\n"
        
        # 按修改時間排序
        sorted_logs = sorted(log_files.items(), key=lambda x: x[1], reverse=True)
        
        for log_file, mtime in sorted_logs[:5]:  # 只顯示最新的5個
            report += f"\n   📄 {os.path.basename(log_file)} (修改: {mtime})"
            
            # 檢查token loss狀態
            if 'token' in log_file.lower() or 'discrete' in log_file.lower():
                status = self.check_token_loss_status(log_file)
                if status:
                    if status['has_token_loss_errors']:
                        report += f" ❌ Token Loss錯誤"
                        if status['latest_error']:
                            report += f" - {status['latest_error']}"
                    
                    if status['current_epoch']:
                        report += f" | Epoch: {status['current_epoch']}"
                    
                    if status['progress']:
                        report += f" | 進度: {status['progress']}"
                        
                    if status['dimension_errors']:
                        unique_errors = list(set(status['dimension_errors']))
                        report += f" | 維度錯誤: {', '.join(unique_errors)}"

        report += f"""

🎯 監控建議:
1. 定期檢查Token Loss實驗是否完成數據準備階段
2. 監控維度匹配修復效果
3. 檢查GPU記憶體使用情況
4. 驗證訓練進度和損失趨勢

📱 使用方法:
   python monitor_experiments.py                    # 單次檢查
   python monitor_experiments.py --continuous      # 持續監控

╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
        return report

    def run_continuous_monitoring(self):
        """持續監控模式"""
        print("🔄 開始持續監控實驗...")
        print(f"⏱️  監控間隔: {self.monitor_interval} 秒")
        print("📞 按 Ctrl+C 停止監控\n")
        
        try:
            while True:
                # 清屏（可選）
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # 生成並顯示報告
                report = self.generate_report()
                print(report)
                
                # 等待下次監控
                time.sleep(self.monitor_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 監控已停止")

def main():
    import sys
    
    monitor = ExperimentMonitor()
    
    if len(sys.argv) > 1 and '--continuous' in sys.argv:
        monitor.run_continuous_monitoring()
    else:
        # 單次檢查
        report = monitor.generate_report()
        print(report)

if __name__ == "__main__":
    main()