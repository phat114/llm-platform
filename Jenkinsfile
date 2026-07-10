// Pipeline deploy "llm-platform" — vLLM + LiteLLM + Open WebUI, trên HOST LAN có GPU.
// KHÔNG build image (stack dùng image công khai: vllm/vllm-openai, litellm, open-webui) →
// deploy = đồng bộ compose/config + `docker compose pull` + `up -d`. Chỉ chạy nhánh `main`.
//
// Chạy trên agent inbound label 'llm-platform' (khai trong repo jenkins-infra/casc.yaml),
// deploy NGAY trên host LAN qua docker.sock. Host yêu cầu: GPU NVIDIA + nvidia-container-toolkit.
//
// ⚠️ .env (secret + model auto-detect) do HOST quản lý tại $CONFIG_DIR, KHÔNG commit (.gitignore)
//    và KHÔNG bị pipeline ghi đè — lần đầu tạo tay trên host (xem stage Deploy). App publish cổng
//    THẲNG ra host LAN (3000 UI / 4000 gateway / 8000 vLLM), không có nginx/reverse proxy.

pipeline {
  agent { label 'llm-platform' }
  options {
    timestamps()
    timeout(time: 30, unit: 'MINUTES')
    disableConcurrentBuilds()      // tránh 2 lần deploy giẫm chân nhau
  }
  environment {
    APP        = 'llm-platform'
    CONFIG_DIR = '/opt/app-config/llm-platform'  // nơi chạy compose (mount CÙNG path host↔agent)
  }
  stages {
    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Test') {
      // Test logic thuần Python (data-mix + eval gate + config loader). KHÔNG cần GPU/torch.
      // Chạy MỌI nhánh (để PR cũng được kiểm), trong venv throwaway. Test router/agents cần
      // openai-agents nên tự SKIP ở đây (deps nhẹ) — chạy được trong venv của agents/.
      steps {
        sh '''
          set -euo pipefail
          python3 -m venv --clear /tmp/llm-test-venv
          . /tmp/llm-test-venv/bin/activate
          pip install -q --disable-pip-version-check pytest pyyaml
          pytest tests/ -q
        '''
      }
    }

    stage('Deploy') {
      when { branch 'main' }
      steps {
        // Deploy qua docker.sock → compose phân giải bind-mount (./scripts, ./config/litellm.yaml)
        // theo path TRÊN HOST, không phải workspace agent. Nên copy file version-controlled xuống
        // $CONFIG_DIR (mount cùng path host↔agent) rồi chạy compose tại đó. .env giữ nguyên.
        sh '''
          set -euo pipefail

          # 1. .env do host quản lý (secret + model auto-detect) — BẮT BUỘC có sẵn, KHÔNG ghi đè.
          install -d "$CONFIG_DIR"
          if [ ! -f "$CONFIG_DIR/.env" ]; then
            echo "‼ Thiếu $CONFIG_DIR/.env — tạo MỘT LẦN trên host rồi chạy lại job:"
            echo "    cp .env.example $CONFIG_DIR/.env"
            echo "    # rồi điền VLLM_API_KEY, LITELLM_MASTER_KEY, HF_TOKEN (nếu model gated)..."
            echo "    # (tùy chọn) dò model 1 lần: cd $CONFIG_DIR && bash scripts/detect-gpu.sh"
            exit 1
          fi

          # 1b. Thư mục adapter/model do HOST quản lý (artifact train — KHÔNG commit/ghi đè), chỉ
          #     đảm bảo TỒN TẠI để bind-mount trong docker-compose.yml không trỏ vào path rỗng.
          #     Adapter thật do pipeline train (training/) đặt vào đây trên host khi bật multi-LoRA.
          install -d "$CONFIG_DIR/training/adapters" "$CONFIG_DIR/training/models"

          # 2. Đồng bộ file version-controlled cần lúc runtime xuống $CONFIG_DIR. .env KHÔNG đụng tới.
          install -m 644 docker-compose.yml "$CONFIG_DIR/docker-compose.yml"
          rm -rf "$CONFIG_DIR/scripts" "$CONFIG_DIR/config"
          cp -a scripts "$CONFIG_DIR/scripts"
          cp -a config  "$CONFIG_DIR/config"

          # 3. Deploy tại $CONFIG_DIR (compose tự nạp .env cùng thư mục). KHÔNG chạy detect-gpu mỗi
          #    lần (GPU host cố định — dò 1 lần lúc setup là đủ); chỉ cập nhật image + rolling up.
          cd "$CONFIG_DIR"
          docker compose pull                       # cập nhật image :latest/:main
          docker compose up -d --remove-orphans
          docker image prune -f
          docker compose ps
        '''
      }
    }

    stage('Health (soft)') {
      when { branch 'main' }
      steps {
        // vLLM tải model lần ĐẦU có thể ~15' (compose start_period 900s), litellm chờ vLLM healthy
        // mới chạy → KHÔNG hard-fail. Chỉ probe nhẹ + hướng dẫn theo dõi. Đọc cổng/key bằng grep
        // (KHÔNG source .env — tránh giá trị nhiều token như VLLM_EXTRA_ARGS làm vỡ shell).
        sh '''
          cd "$CONFIG_DIR" || exit 0
          PORT="$(grep -E '^LITELLM_PORT=' .env | tail -1 | cut -d= -f2)"; PORT="${PORT:-4000}"
          KEY="$(grep -E '^LITELLM_MASTER_KEY=' .env | tail -1 | cut -d= -f2-)"
          if curl -sf "http://localhost:${PORT}/v1/models" -H "Authorization: Bearer ${KEY}" >/dev/null 2>&1; then
            echo "[health] ✅ gateway sẵn sàng (http://<host-LAN>:${PORT}/v1)"
          else
            echo "[health] ⏳ gateway chưa sẵn — vLLM có thể còn đang tải model (bình thường ở lần đầu)."
            echo "         theo dõi tải model: docker compose -f $CONFIG_DIR/docker-compose.yml logs -f vllm"
            echo "         kiểm tra đầy đủ   : cd $CONFIG_DIR && bash scripts/healthcheck.sh"
          fi
        '''
      }
    }
  }
  post {
    success { echo "✅ ${APP}: deployed lên host LAN (#${BUILD_NUMBER}) — UI :3000 · gateway :4000 · vLLM :8000" }
    failure { echo "❌ ${APP}: deploy FAILED" }
  }
}
