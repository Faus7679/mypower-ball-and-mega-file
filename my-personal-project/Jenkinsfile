pipeline {
  agent any

  triggers {
    githubPush()
  }

  options {
    disableConcurrentBuilds()
    timestamps()
  }

  environment {
    PROJECT_NAME = 'my-personal-project'
    AWS_ACCOUNT_ID = '382975714575'
    AWS_REGION = 'us-east-1'
    TERRAFORM_VERSION = '1.7.5'
    PATH = "${WORKSPACE}/bin:${env.PATH}"
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Install Terraform') {
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          mkdir -p "$WORKSPACE/bin"

          if ! command -v terraform >/dev/null 2>&1; then
            curl -fsSL "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip" -o /tmp/terraform.zip
            unzip -qo /tmp/terraform.zip -d "$WORKSPACE/bin"
            chmod +x "$WORKSPACE/bin/terraform"
          fi

          terraform version
        '''
      }
    }

    stage('Terraform Format') {
      steps {
        sh 'terraform fmt -check -recursive'
      }
    }

    stage('Terraform Init') {
      steps {
        withCredentials([
          string(credentialsId: 'aws-access-key-id', variable: 'AWS_ACCESS_KEY_ID'),
          string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
        ]) {
          sh 'terraform init'
        }
      }
    }

    stage('Terraform Validate') {
      steps {
        withCredentials([
          string(credentialsId: 'aws-access-key-id', variable: 'AWS_ACCESS_KEY_ID'),
          string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
        ]) {
          sh 'terraform validate'
        }
      }
    }

    stage('Terraform Plan') {
      steps {
        withCredentials([
          string(credentialsId: 'aws-access-key-id', variable: 'AWS_ACCESS_KEY_ID'),
          string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
        ]) {
          sh '''#!/usr/bin/env bash
            set -euo pipefail
            terraform plan \
              -input=false \
              -out=tfplan \
              -var="project_name=${PROJECT_NAME}" \
              -var="aws_region=${AWS_REGION}" \
              -var="aws_account_id=${AWS_ACCOUNT_ID}"
          '''
        }
      }
    }

    stage('Terraform Apply') {
      when {
        anyOf {
          branch 'dev'
          branch '*'
          branch 'main'
          branch 'master'
        }
      }
      steps {
        withCredentials([
          string(credentialsId: 'aws-access-key-id', variable: 'AWS_ACCESS_KEY_ID'),
          string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
        ]) {
          sh 'terraform apply -input=false -auto-approve tfplan'
        }
      }
    }
  }
}