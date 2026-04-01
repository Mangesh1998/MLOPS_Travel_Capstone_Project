pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'docker build -t flight-price-api .'
            }
        }
        stage('Test') {
            steps {
                sh 'docker run flight-price-api python -c "import app; print(\"Test passed\")"'
            }
        }
        stage('Deploy') {
            steps {
                sh 'kubectl apply -f k8s_deployment.yaml'
            }
        }
    }
}