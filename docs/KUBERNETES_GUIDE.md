# Kubernetes Production Orchestration Guide

## Overview
This document covers applying Kubernetes manifests (`k8s/`), configuring autoscaling, persistent volumes, configmaps, and ingress controllers.

## Manifest Directory (`k8s/`)
- `deployment-api.yaml`: API deployment (RollingUpdate, 3 replicas).
- `deployment-worker.yaml`: Background queue processing worker.
- `deployment-dashboard.yaml`: Streamlit frontend.
- `service-api.yaml` & `service-dashboard.yaml`: ClusterIP services.
- `horizontal-pod-autoscaler.yaml`: Autoscaling based on CPU (75%) & Memory (80%).
- `persistent-volume.yaml`: PVC for artifacts and logs.
- `ingress.yaml`: Ingress controller configuration.

## Deployment Commands
```bash
# Apply ConfigMaps and Secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/persistent-volume.yaml

# Deploy Workloads and Services
kubectl apply -f k8s/service-api.yaml
kubectl apply -f k8s/service-dashboard.yaml
kubectl apply -f k8s/deployment-api.yaml
kubectl apply -f k8s/deployment-worker.yaml
kubectl apply -f k8s/deployment-dashboard.yaml

# Enable Ingress and HPA
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/horizontal-pod-autoscaler.yaml

# Check Rollout Status
kubectl rollout status deployment/cip-api
```
