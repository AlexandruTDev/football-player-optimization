# Football Player Optimisation 

* Machine Learning algortihm that calculates and predicts players injury risk based on data gathered in matches and training

  # Project structure:
    * The ML module uses raw data from the data/raw folder (in csv format)
      * The raw data contains entries gathhered from Catapult
    * Generated visuals can be found in visualisations folder
      * Injury risk maps & Confussion matrix 

  # Current project status:
    * Able to parse raw data and train ML models using various prediction algorithms (RandomForestClassifier, GradientBoosting & NeuralNetwork -> MLPClassifier)
      * The prediction algortihms have been selected in order to compare results based on their general performance metrics for certain jobs.
    * Create visuals based on available data
      * These can be found in the designated visualisations folder of the project
    * Code base has been developed in PyCharm & FirebaseStudio using AI tools for refinment and suggestions (Claude AI - Sonet 4.0 & Gemini 2.5 Flash)
 
  # Running the project:
    * Import github project into IDE
    * Create temporary work environment by running following command into terminal:
      * nix-shell dev.nix
    * Run project pipeline in order to process raw data, train models and generate associated viusal files
      * python src/run_pipeline.py --data data/raw/report_export.csv

  # Roadmap:
    * Split available data into training & test data
    * Evaluate Machine Learning Model based on confussion matrix
    * Optimise module based on results
    * Extend model capacity to predicting best attacking patterns for a team
