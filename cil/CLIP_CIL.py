import torch
import torch.nn as nn
from torchvision import datasets, transforms
from continuum.datasets import PyTorchDataset
from continuum.tasks import split_train_val, TaskSet
import clip as CLIP
from sklearn.cluster import KMeans


# Define the incremental learning parameters
NUM_TASKS = 10
BATCH_SIZE = 32

# Load the CIFAR-100 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
cifar100_train = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
cifar100_test = datasets.CIFAR100('./data', train=False, download=True, transform=transform)

# Split the dataset into tasks
train_tasks, val_tasks = split_train_val(cifar100_train, NUM_TASKS)

# Define the CLIP model with a learnable prompt pool
clip_model = CLIP(clip_model='ViT-B/32', prompt_encoder=True, normalize=False)
prompt_pool = nn.Parameter(torch.randn(NUM_TASKS, clip_model.context_length, clip_model.dim), requires_grad=True)
clip_model.add_module('prompt_pool', prompt_pool)

# Define the optimizer for the prompt pool
optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-3)

# Train the learnable prompt pool
clip_model.train()
for task in train_tasks:
    train_dataset = PyTorchDataset(task)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = clip_model(inputs, labels, task_id=task.id)['loss']
        loss.backward()
        optimizer.step()

# Test the model using K-Means clustering to match inputs to prompts
clip_model.eval()
with torch.no_grad():
    for task in val_tasks:
        test_dataset = PyTorchDataset(task)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        task_set = TaskSet(task)

        prompt_embeddings = []
        for i in range(NUM_TASKS):
            prompt_embedding = clip_model.encode_text(prompt_pool[i], task_id=i)
            prompt_embeddings.append(prompt_embedding)
        
        for inputs, labels in test_loader:
            input_embeddings = clip_model.encode_image(inputs)

            # Use K-Means to match input embeddings to prompt embeddings
            kmeans = KMeans(n_clusters=NUM_TASKS)
            kmeans.fit(input_embeddings.cpu().numpy())
            task_ids = kmeans.labels_

            # Evaluate the model for each task
            for i in range(NUM_TASKS):
                task_i_indices = torch.where(task_ids == i)[0]
                task_i_inputs = inputs[task_i_indices]
                task_i_labels = labels[task_i_indices]
                task_i_set = task_set.get_sub_task(i)

                task_i_logits = clip_model(task_i_inputs, task_i_labels, task_id=i)['logits']
                task_i_acc = (task_i_logits.argmax(dim=-1) == task_i_labels).float().mean().item()
                print(f"Task {i} accuracy: {task_i_acc:.4f}")
