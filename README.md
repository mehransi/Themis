# pelastic


## Instructions
1. Create a Kubernetes cluster (on all cluster nodes)
    1. Install Microk8s: [Get started](https://microk8s.io/docs/getting-started) (for channel, use `--channel=1.30/edge`)
    2. microk8s enable dns
    <!-- 3. microk8s enable observability -->
    3. microk8s stop
    4. Edit the files `kube-apiserver`, `kubelet`, `kube-scheduler`, `kube-controller-manager` in the directory `/var/snap/microk8s/current/args/` to make sure `--feature-gates=InPlacePodVerticalScaling=true` is added to all of them.
    5. microk8s start

2. on Master node run: `microk8s add-node`
3. Copy the join command (with --worker) and run it on worker nodes
