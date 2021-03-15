## Predicting Life Expectancy using Machine Learning
Project Summary Our aim is to predict the life expectancy of a person of a given country considering a number of factors like sex differences, mental illnesses, etc. The data set provides us with a number of factors which could be considered to make this prediction.

Project Requirements The project requires the data set, Python, IBM Watson Studio, IBM Cloud as well as Node Red.

Functional Requirements The project is required to make the prediction of life expectancy. Moreover, it has perform data pre-processing and data analysis to make sure that accuracy could be achieved.

Technical Requirements The programming language used to make the project would be Python. The front end would be done on Noe Red which is a tool provided by IBM. Alos, IBM cloud would we used to deploy the project online and make it available for people all around the world.

Software Requirements A stable internet connection is all that is needed to access the project. Users will just need to enter the URL of the website hosted on IBM Cloud to access the entire project.

Project Deliverables The project will have a good user interface so that users can find it easy to interact to. Moreover, it will predict the life expectancy of the person who is living in a particular country. The user will have to enter a number of details so that the model can calculate according and predict the life expectancy of a person with those details.

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. model.py - This contains code fot our Machine Learning model to predict employee salaries absed on trainign data in 'hiring.csv' file.
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. templates - This folder contains the HTML template to allow user to enter employee detail and displays the predicted employee salary.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You should be able to view the homepage as below :
![alt text](http://www.thepythonblog.com/wp-content/uploads/2019/02/Homepage.png)

Enter valid numerical values in all 3 input boxes and hit Predict.

If everything goes well, you should  be able to see the predcited salary vaule on the HTML page!
![alt text](http://www.thepythonblog.com/wp-content/uploads/2019/02/Result.png)

4. You can also send direct POST requests to FLask API using Python's inbuilt request module
Run the beow command to send the request with some pre-popuated values -
```
python request.py
```
