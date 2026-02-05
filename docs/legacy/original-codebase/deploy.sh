#!/bin/bash


function pretty_title(){
  printf '=%.0s' $(seq -3 ${#1})
  printf '\n= %s =\n' "${1}"
  printf '=%.0s' $(seq -3 ${#1})
  printf '\n'
}

pretty_title "Deployer"
echo $(whoami)
echo $HOSTNAME
/snap/bin/docker ps
rm -rf filos
pretty_title "Cloning the repo"
git clone git@bitbucket.org:filos-crypto/filos.git
cd filos/ || exit
export IP=$(hostname -I | cut -d ' ' -f 1)
echo "REACT_APP_REST_API_URL=http://${IP}:5050" > frontend/filosui/.env
ls -lah frontend/filosui
/snap/bin/docker system prune -f
/snap/bin/docker images
pretty_title "Building new docker images"
/snap/bin/docker-compose build
pretty_title "Stopping old version of the app"
/snap/bin/docker-compose down
pretty_title "Starting new version of the app"
/snap/bin/docker-compose up -d


