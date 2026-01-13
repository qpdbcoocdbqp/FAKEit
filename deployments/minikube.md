# Use Minikube to run pods with GPU resource on Windows

## Prerequirents

* **wsl**
    ```sh
    wsl --update
    ```

In `wsl`,

* **nvidia-container-toolkit**

    ```sh
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get install -y nvidia-container-toolkit
    nvidia-ctk --version
    ```

* **kubectl**

    Ensure `kubectl` is installed.

* **minikube**

    ```sh
    curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
    sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
    ```


## Run `minikube`

```sh

minikube start --driver docker --container-runtime docker --gpus all --memory=8192 --cpus=8
kubectl describe node minikube
kubectl get po --all-namespaces

# NAMESPACE     NAME                                   READY   STATUS    RESTARTS      AGE
# kube-system   coredns-66bc5c9577-vzdzx               1/1     Running   0             115s
# kube-system   etcd-minikube                          1/1     Running   0             2m1s
# kube-system   kube-apiserver-minikube                1/1     Running   0             2m1s
# kube-system   kube-controller-manager-minikube       1/1     Running   0             2m1s
# kube-system   kube-proxy-vsc5q                       1/1     Running   0             115s
# kube-system   kube-scheduler-minikube                1/1     Running   0             2m1s
# kube-system   nvidia-device-plugin-daemonset-g892d   1/1     Running   0             115s
# kube-system   storage-provisioner                    1/1     Running   1 (91s ago)   119s

kubectl logs -f nvidia-device-plugin-daemonset-g892d -n kube-system
# ...
# 111 06:10:54.221954       1 server.go:195] Starting GRPC server for 'nvidia.com/gpu'
# I0111 06:10:54.222586       1 server.go:139] Starting to serve 'nvidia.com/gpu' on /var/lib/kubelet/device-plugins/nvidia-gpu.sock
# I0111 06:10:54.224783       1 server.go:146] Registered device plugin for 'nvidia.com/gpu' with Kubelet

kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# NAME       GPU
# minikube   1

kubectl apply -f deployments/test/gpu-verification.yaml
kubectl get pods gpu-verification

# NAME               READY   STATUS      RESTARTS   AGE
# gpu-verification   0/1     Completed   0          5s

kubectl logs pods/gpu-verification

# Sun Jan 11 06:17:43 2026       
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 580.105.07             Driver Version: 581.80         CUDA Version: 13.0     |
# ...

kubectl delete pods gpu-verification
# pod "gpu-verification" deleted from default namespace

```

## Run deployment

* **Not recommand to use minikube to run on Windows. Because image pull to minikube is slow.**
* **It waiting so long. I did not success yet.**

```sh
kubectl apply -k deployments/base
```
