dependency:
	curl --silent --location https://dl.yarnpkg.com/rpm/yarn.repo | sudo tee /etc/yum.repos.d/yarn.repo
	sudo rpm --import https://dl.yarnpkg.com/rpm/pubkey.gpg
	sudo yum install -y yarn
	yarn add @tensorflow-models/pose-detection
	yarn add @tensorflow/tfjs-core
	yarn add @tensorflow/tfjs-converter
	yarn add @tensorflow/tfjs-backend-webgl
	npm install compute-cosine-similarity
	npm install parcel-bundler --save-dev

run:
	./node_modules/.bin/parcel index.html

open:
	open http://localhost:1234?model=movenet
