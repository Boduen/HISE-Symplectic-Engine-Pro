HISE-Pro 啟動與測試標準作業程序 (SOP)
本指南涵蓋 HISE-Pro 物理感知 AI 模型 的環境建置、單元測試、整合測試與訓練啟動步驟。
1. 安裝環境 (Installation)
前置要求
 * 作業系統：Linux 或 Windows (WSL2)
 * Python 版本：>= 3.9
 * CUDA 版本：>= 12.1
步驟 1：安裝 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

步驟 2：安裝核心依賴 (Triton & Transformers)
pip install transformers triton einops wandb

步驟 3：安裝工具庫
pip install pandas matplotlib seaborn streamlit scipy

2. 單元測試：物理內核 (Unit Test: Kernels)
建立一個名為 test_kernel.py 的檔案，用於驗證 Triton Fused Kernel 是否能與 GPU 正常通訊。
執行測試指令：
python test_kernel.py

3. 整合測試：模型前向傳播 (Integration Test)
驗證 Config、MoE Router 與 Base Layers 的組裝正確性。
執行指令：
python 9.test_run.py

驗證檢核點 (Checklist)
 * [ ] 終端機顯示 Inference Time: XX.XX ms
 * [ ] 終端機顯示 [PASS] FSI Metric Captured
 * [ ] 若啟用 MoE，確認無 RuntimeError
4. 啟動訓練 (Start Training)
執行小規模預訓練以驗證物理損失函數 (Symplectic Fisher Loss) 的收斂性。
執行指令 (單卡除錯模式)
# 設定環境變數以模擬單機單卡
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export WANDB_MODE=disabled 

# 啟動訓練腳本
python 13.train_pretrain_distributed.py

觀察指標
 * Loss 下降：觀察 Loss 是否穩定下降（例如 10.x -> 9.x）。
 * 物理權重：觀察 loss_physics 是否非零（代表 FSI 機制生效）。
 * 運行速度：確認 Iter/s 速度正常，無明顯卡頓。
5. 常見錯誤排除 (Troubleshooting)
| 錯誤訊息 | 可能原因 | 解決方法 |
|---|---|---|
| ImportError: cannot import name 'triton' | 環境未安裝 Triton | 確認在 Linux/WSL2 環境下，執行 pip install triton |
| RuntimeError: stack expects a non-empty Tensor... | MoE 模式 FSI 為空 | 檢查 6.hise_modeling_modeling_hise.py 是否已加入空值檢查修復 |
| Loss is NaN | 數值梯度爆炸 | 確認 2.hise_thermodynamics_mass_dynamics.py 已使用 Softplus |
| CUDA error: device-side assert triggered | 詞彙表索引越界 | 檢查 config.vocab_size 是否匹配輸入數據 |
