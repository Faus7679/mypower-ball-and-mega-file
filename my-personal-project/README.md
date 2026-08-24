# my-personal-project

Terraform infrastructure project with CI/CD definitions for Azure DevOps, Jenkins, and GitHub Actions.

## AWS deployment target

- AWS account ID: `382975714575`
- Project name: `my-personal-project`
- Default AWS region: `us-east-1`

## GitHub Actions pipeline

The GitHub Actions workflow is defined in `.github/workflows/terraform-cicd.yml`.

- Trigger: every push to the GitHub repository
- Manual run: supported through `workflow_dispatch`
- Behavior:
  - Runs `terraform fmt`, `terraform init`, `terraform validate`, and `terraform plan` on every push
  - Runs `terraform apply` automatically when the push is to `main` or `master`

Required GitHub secret:

- `AWS_ROLE_ARN`: IAM role ARN in account `382975714575` that GitHub Actions can assume using OIDC

Recommended role name example:

- `arn:aws:iam::382975714575:role/github-actions-terraform`

## Jenkins pipeline

The Jenkins pipeline is defined in `Jenkinsfile`.

- Trigger: GitHub push webhook via `githubPush()`
- Behavior:
  - Checks out the repo
  - Installs Terraform if it is not already available on the Jenkins agent
  - Runs `terraform fmt`, `terraform init`, `terraform validate`, and `terraform plan` on every push
  - Runs `terraform apply` automatically when the pushed branch is `main` or `master`

Required Jenkins credentials:

- `aws-access-key-id`
- `aws-secret-access-key`

To trigger Jenkins on new commits pushed to GitHub, configure the repository webhook to point to your Jenkins webhook endpoint:

- `https://<your-jenkins-host>/github-webhook/`

## Terraform notes

The Terraform configuration now:

- Restricts execution to AWS account `382975714575`
- Uses project-based resource naming with `my-personal-project`
- Resolves the EC2 AMI dynamically instead of using a stale hard-coded AMI
- Uses an IAM instance profile correctly for the EC2 instance
