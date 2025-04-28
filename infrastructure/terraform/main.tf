provider "aws" {
  region = var.aws_region
}

# EKS Cluster
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = var.cluster_name
  cluster_version = "1.24"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Node groups configuration
  eks_managed_node_groups = {
    gpu = {
      name = "gpu-node-group"
      
      instance_types = ["g4dn.xlarge"]
      capacity_type  = "ON_DEMAND"
      
      min_size     = 1
      max_size     = 5
      desired_size = 2
      
      labels = {
        Environment = "production"
        NodeType    = "gpu"
      }
      
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
    
    general = {
      name = "general-node-group"
      
      instance_types = ["t3.large"]
      capacity_type  = "SPOT"
      
      min_size     = 2
      max_size     = 10
      desired_size = 3
      
      labels = {
        Environment = "production"
        NodeType    = "general"
      }
    }
  }
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  single_nat_gateway = false
  
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Environment = "production"
    Project     = var.project_name
  }
}

# Redis ElastiCache
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.project_name}-redis"
  engine              = "redis"
  node_type           = "cache.t3.medium"
  num_cache_nodes     = 1
  parameter_group_name = "default.redis6.x"
  port                = 6379
  security_group_ids  = [aws_security_group.redis.id]
  subnet_group_name   = aws_elasticache_subnet_group.redis.name
}

# Security Groups
resource "aws_security_group" "redis" {
  name        = "${var.project_name}-redis-sg"
  description = "Security group for Redis cluster"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }
}

# S3 Bucket for model artifacts
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${var.project_name}-model-artifacts"
  acl    = "private"
  
  versioning {
    enabled = true
  }
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}