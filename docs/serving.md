# Deploying the serving system

## Backend

Set up Docker Swarm.

```sh
docker swarm init
docker swarm join
```

Create a custom swarm overlay network.

```sh
docker network create --attachable -d overlay leaderboard
```

Spawn the workers with `deployment/tgi-container.sh`


## Install our TGI (text generation inference) fork

## Start workers
Should read from deployment.yaml. Here we just start one worker manually. 
Add `-d` to detach container. 

```commandline
GPU_ID=0
PORT=8001
docker rm worker0
docker run  --name=worker0  --network=mynetwork --gpus '"device='"$GPU_ID"'"' --shm-size 1g -p $PORT:8000 -v /data/leaderboard/tgi-data:/data -v /data/leaderboard/weights:/weights ml-energy/tgi:latest --model-id bigscience/bloom-560m --port 8000
```

```commandline
GPU_ID=1
PORT=8002
docker rm worker1
docker run  --name=worker1  --network=mynetwork --gpus '"device='"$GPU_ID"'"' --shm-size 1g -p $PORT:8000 -v /data/leaderboard/tgi-data:/data -v /data/leaderboard/weights:/weights ml-energy/tgi:latest --model-id tiiuae/falcon-7b-instruct  --port 8000
```

More models is coming soon.

## Start controller and web server
```commandline
port=8000
docker_port=8000
docker run --name=controller -v $HOME/leaderboard:/code  --network=mynetwork -p 8000:8000 -it  tgi:origin
```

```commandline
python controller.py --host 0.0.0.0
python app.py --share
```

The controller will check the live workers every min.
 