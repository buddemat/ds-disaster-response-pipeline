# Disaster Response Pipeline Project
This project contains my project submission to the Disaster Response Pipeline project of the Udacity Data Scientist course.

## Execution

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
   ```
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```
    - To run ML pipeline that trains classifier and saves
   ```
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```

1. Run the following command in the app's directory to run your web app.
   ```
   python run.py
   ```

1. Go to http://0.0.0.0:3001/

## Documentation

The documentation of this project was built using the Python Documentation Generator [Sphinx](https://www.sphinx-doc.org/). To do so, execute the following commands:

1. Install Sphinx (only once):
   ```
   pip install -U sphinx
   ```

1. Generate documentation:
   ```
   sphinx-build -b html docs/source/ docs/build/html
   ```
   or
   ```
   cd docs/source
   make html
   ```
