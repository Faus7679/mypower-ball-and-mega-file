# main.tf

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "project_name" {
  description = "Project name used for tagging and naming resources."
  type        = string
  default     = "my-personal-project"
}

variable "aws_region" {
  description = "AWS region where resources will be deployed."
  type        = string
  default     = "us-east-1"
}

variable "aws_account_id" {
  description = "Expected AWS account ID for Terraform deployments."
  type        = string
  default     = "382975714575"
}

locals {
  common_tags = {
    Project   = var.project_name
    ManagedBy = "Terraform"
  }

  bucket_name = "${var.project_name}-${var.aws_account_id}-assets"
}

provider "aws" {
  region              = var.aws_region
  allowed_account_ids = [var.aws_account_id]
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }

  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_vpc" "personal_vpc" {
  cidr_block = "10.0.0.0/16"

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-vpc"
  })
}

resource "aws_s3_bucket" "personal_bucket" {
  bucket = local.bucket_name

  tags = merge(local.common_tags, {
    Name = local.bucket_name
  })
}

resource "aws_iam_role" "admin_role" {
  name = "${var.project_name}-admin-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "admin_attach" {
  role       = aws_iam_role.admin_role.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

resource "aws_iam_policy" "admin_policy" {
  name        = "${var.project_name}-admin-policy"
  description = "Policy for admin role to manage EC2 and VPC resources"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ec2:CreateVpc",
        "ec2:DescribeVpcs",
        "ec2:CreateSubnet",
        "ec2:DescribeSubnets",
        "ec2:CreateTags",
        "ec2:RunInstances",
        "ec2:DescribeInstances"
      ]
      Resource = "*"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_instance_profile" "admin_profile" {
  name = "${var.project_name}-instance-profile"
  role = aws_iam_role.admin_role.name
}

resource "aws_subnet" "personal_subnet" {
  vpc_id            = aws_vpc.personal_vpc.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "${var.aws_region}a"

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-subnet"
  })
}

resource "aws_instance" "personal_instance" {
  ami                  = data.aws_ami.amazon_linux.id
  instance_type        = "t2.micro"
  subnet_id            = aws_subnet.personal_subnet.id
  iam_instance_profile = aws_iam_instance_profile.admin_profile.name

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-instance"
  })
}

output "vpc_id" {
  value = aws_vpc.personal_vpc.id
}

output "s3_bucket_name" {
  value = aws_s3_bucket.personal_bucket.bucket
}

output "iam_role_name" {
  value = aws_iam_role.admin_role.name
}

output "ec2_instance_id" {
  value = aws_instance.personal_instance.id
}
