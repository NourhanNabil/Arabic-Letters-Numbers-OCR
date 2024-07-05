## Dataset
Arabic Letter and Numbers Dataset Consist of Images of Arabic Number and letters. You can find the dataset [here](https://www.kaggle.com/datasets/mahmoudreda55/arabic-letters-numbers-ocr).

## Clone the repository
```bash
git clone https://github.com/NourhanNabil/face-verification.git
cd face-verification
```

### Prerequisites
- Docker installed on your machine.
- Docker Compose installed on your machine.


**Docker**
Build and Run the Docker Containers:
```bash
docker-compose up -d
```

**Interactive Shell**
For interactive shell access while the container is running, you can use:
```bash
docker-compose exec app bash
```

**Access the App**
Open your web browser and go to `http://localhost:8000` to access the App.

**Shut Down the Containers**
```bash
docker-compose down # Stops and removes containers, networks, volumes, and other services.
docker-compose stop # Stops containers without removing them, allowing you to start them again later.
```