PROJECT_ID=red-seeker-334115
#make deploy
deploy:
	gcloud beta functions deploy event-new-prediction-created \
		--project ${PROJECT_ID} \
		--runtime python39 \
		--trigger-topic topic-new-prediction-created \
		--entry-point initial_method \
		--memory 512MB \
		--timeout 50s