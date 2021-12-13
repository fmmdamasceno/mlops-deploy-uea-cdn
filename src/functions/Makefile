PROJECT_ID=mlops-uea-2
#make deploy
deploy:
	gcloud beta functions deploy event-new-score-created \
		--project ${PROJECT_ID} \
		--runtime python39 \
		--trigger-topic topic-new-score-created \
		--entry-point initial_method \
		--memory 512MB \
		--timeout 50s