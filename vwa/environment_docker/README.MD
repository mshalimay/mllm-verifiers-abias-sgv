# README
This file contains instructions to:
- Set up the environments for (Visual)WebArena evaluation (see A or B)
- Start, reset, check status of websites manually (optional; see C, D)
- Clean leftovers from docker containers manually (optional; see E)
- Changes needed when hosting websites in different machines (optional; see F)

The scripts `download_load_envs.py` or `download_load_envs.sh` automate the download and loading of the the docker containers required for each website in (Visual)WebArena. However, if you prefer to set the containers manually, follow the instructions in (B).

Original instructions available at: [link](//github.com/web-arena-x/webarena/tree/main/environment_docker).

# A) Download and Load containers via the automated scripts
To use the automated scripts:

```bash
cd vwa
python -m scripts.environments.ensure_docker_imgs <site1> <site2> ...
# or
# scripts/environments/download_load_envs.sh <site1> <site2> ...
```
Where site in `['shopping' 'admin' 'reddit' 'gitlab' 'classifieds' 'wikipedia' 'all_vwa' 'all_wa']`. The special options `all_vwa` and `all_wa` download and loads all data needed for `VisualWebArena` and `WebArena`, respectively.

---

# B) Download and Load containers manually.
The following provides instructions to:
1) Download the necessary files to deploy the (Visual)WebArenas via docker containers.
2) Load images into Docker for the websites that requires it.
3) Start/Reset the websites manually (optional).


## 1. Download necessary files for local deployment

The sections below contain the links to download the files necessary to deploy each website.

Save them in `vwa/environment_docker/docker_imgs` or change `DOCKER_IMGS_PATH` in `vwa/benchmark_config/constants` and save in your preferred directory.

**Notes**:
- Files for each website are provided in different formats:
    - `.tar`: These are compressed Docker images, which are loaded into the Docker image cache via the `docker load` command.
    - `.zim`: Can run Docker directly on top of it; no need to load to Docker image cache.
    - Docker-Compose: The image is loaded into the Docker image cache directly from Docker's remote repository.

- Each benchmark requires the following set of websites for full evaluation:
    - **VWebArena**: [Shopping, Classifieds, Reddit, Wikipedia, Homepage]
    - **WebArena**: [Shopping, Shopping Admin, GitLab Reddit, Wikipedia, Homepage]
    - Obs: Running websites in isolation is possible and useful for testing, however:
        - (i)  Tasks that require navigating to more than one website will be impossible to complete.
        - (ii) The agent will not be able to use tools (as in the homepage) or knowledge source (Wikipedia).
        
### Links to download the docker containers:
#### Shopping (~63GB) [VWebArena, WebArena]
- https://drive.google.com/file/d/1gxXalk9O0p9eu1YkIJcmZta1nvvyAJpA/view?usp=sharing
- http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar

#### Shopping Admin (~9GB) [WebArena]
- https://drive.google.com/file/d/1See0ZhJRw0WTTL9y8hFlgaduwPZ_nGfd/view?usp=sharing
- http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar

#### Reddit (~50GB) [VWebArena, WebArena]
- https://drive.google.com/file/d/17Qpp1iu_mPqzgO_73Z9BnFjHrzmX9DGf/view?usp=sharing
- http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar

#### GitLab (~72GB) [WebArena]
- https://drive.google.com/file/d/19W8qM0DPyRvWCLyQe0qtnCWAHGruolMR/view?usp=sharing
- http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar

#### Wikipedia (~89GB) [WebArena, VWebArena]
- https://drive.google.com/file/d/1Um4QLxi_bGv5bP6kt83Ke0lNjuV9Tm0P/view?usp=sharing
- http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim

#### Classifieds (~81MB) [VWebArena]
(i) Files to compose the containers
- https://drive.google.com/file/d/1m79lp84yXfqdTBHr6IS7_1KkL4sDSemR/view
- https://archive.org/download/classifieds_docker_compose

(ii) Unzip the files in `vwa/environment_docker/docker_imgs`

(iii) Run the below command to download Docker image from remote Docker registry:
```
docker pull jykoh/classifieds:latest
docker pull mysql:8.1
```

### Map [WebArena]
- Coming soon.


## 2. Load images into Docker
The script `scripts/environments/load_docker_imgs.sh` loads the images downloaded in step 1 into the docker image cache. Run:
- `./scripts/environments/load_docker_imgs.sh <site_name>`

Where site_name in: `['shopping' 'admin' 'reddit' 'gitlab' 'classifieds' 'wikipedia' 'all_vwa' 'all_wa']`

**NOTES**: 
- This step take a couple minutes for each website.
- Needs to **run only one time for each website**. Once image is loaded into Docker, you can deploy, delete, stop the containers whenever needed. 
- Run `docker images` to see which ones are loaded.
- For `classifieds`, script runs `docker pull ...` as in part 1. If didn't download the image in step (1), the script will download the whole image to load into docker, and this will take a while.
- After loading images into Docker, you can delete `.tar` files to reclaim storage if desired.

### Verify execution, delete images
After execution, you can run the command below to check which images were sucessfully loaded into Docker: 

```shell
docker images
```

If any image was already loaded, you can reload it by deleting them from cache and running the script again:

```shell
docker image list  # get the image ID; suppose it is `0dbbdef09dd9` for shopping website:
docker rmi 0dbbdef09dd9
./scripts/environments/load_docker_imgs.sh shopping
```

# C) Start/Reset environments manually (optional)
**NOTE**: needs to complete step (A) or (B) for all websites, except `homepage`.

We modified the codebase so environments are started/resetted automatically when running experiments.

However, if you want to start/reset environments manually, you can do so as follows:

1. Run `scripts/environments/start_reset_envs.sh` or `scripts/environments/start_reset_envs.py` as follows:
```
scripts/environments/start_reset_envs.sh [-p] [environment_id] <site_name1> <site_name2> ...

python -m scripts.environments.start_reset_envs [-p] [environment_id] <site_name1> <site_name2> ...
```

Where:
- site_name in: `['shopping' 'admin' 'reddit' 'gitlab' 'classifieds' 'wikipedia' 'homepage']` or `all_vwa` (for all VWebArena sites) or `all_wa` (for all WebArena sites).
- `-p` is an integer ID for the set of websites. Used mostly in parallel execution to keep environments isolated. 

2. The scripts will print a list of docker containers running and the status of each of the websites (i.e., if they are reachable or not)

# D) Checking status of websites (optional)
`./scripts/environments/start_reset_envs.sh` already prints a check on the websites.

You can also check the status of websites through the following options:

- Run `./scripts/environments/check_websites.sh [environment_id]` to check which websites are reachable. If `env_id` is not provided, checks all environments running.

- Open `http://<BASE_URL>:<port>` in a browser, or run `curl -I http://<BASE_URL>:<port>` to quickly check it responds.

# E) Cleaning leftovers (optional)
The runner scripts clean leftovers, but if you can clean them manually by:

```
./scripts/environments/clean_containers.sh [environment_id] 
# if no env_id provided: identify all running environments, asks for confirmation and clean
```

# F) Accessing via other machines on the same network
If you are in shared cluster, you can access a website hosted in `machine_a` through any other machines by substituting 'localhost' by  `machine_a` IP. 

Example:
Suppose you did all the steps above in `machine_a`, and the `reddit` website is available at `http://localhost:9999` in `machine_a`.

To access this website via `machine_b`, use:  `http:/<machine_b_ip>:9999`
