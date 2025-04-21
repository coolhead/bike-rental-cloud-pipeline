# Makefile for Bike Rental Cloud Pipeline 🚲☁️

build:
	docker compose build

up:
	docker compose up --build

up-detached:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

stop:
	docker compose stop

restart:
	docker compose restart

clean:
	docker system prune -af --volumes

# Bonus Targets
test-api:
	curl http://localhost:8000/

test-predict:
	curl -X POST "http://localhost:8000/predict" \
		-H "Content-Type: application/json" \
		-d '{"season":1,"yr":0,"mnth":1,"holiday":0,"weekday":6,"workingday":0,"weathersit":2,"temp":14.110847,"atemp":18.18125,"hum":80.5833,"windspeed":10.749882}'

