.PHONY: up down restart logs health detect pull

up:            ## Auto-detect GPU + khởi động toàn bộ
	bash scripts/start.sh

down:          ## Dừng toàn bộ
	bash scripts/stop.sh

restart: down up ## Khởi động lại

logs:          ## Xem log vLLM (tải model / lỗi)
	docker compose logs -f vllm

health:        ## Kiểm tra 3 tầng
	bash scripts/healthcheck.sh

detect:        ## Chỉ chạy auto-detect model (ghi .env)
	bash scripts/detect-gpu.sh

pull:          ## Kéo image mới nhất
	docker compose pull
