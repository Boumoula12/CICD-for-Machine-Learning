install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md

	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md

	cml comment create report.md

# 🌿 Sauvegarder les résultats dans la branche "update"
update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

# 🔐 Login Hugging Face + upload vers le Space
hf-login:
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	# App (fichiers Gradio : iris_app.py, README.md, requirements.txt)
	huggingface-cli upload Abdelouafi/iris-classification-cicd ./App \
		--repo-type=space --commit-message="Sync App files"

	# Modèle
	huggingface-cli upload Abdelouafi/iris-classification-cicd ./Model /Model \
		--repo-type=space --commit-message="Sync Model"

	# Résultats (métriques + confusion matrix)
	huggingface-cli upload Abdelouafi/iris-classification-cicd ./Results /Metrics \
		--repo-type=space --commit-message="Sync Metrics"

deploy: hf-login push-hub
