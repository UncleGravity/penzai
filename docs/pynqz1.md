Bringup:
```sh
# Load sd card image

# Copy key: 
ssh-copy-id xilinx@pynq

# Conect: 
ssh xilinx@pynq

# Remove sudo password req:
echo "$USER ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/$USER > /dev/null

# To revert in the future:
# sudo rm /etc/sudoers.d/$USER


```
