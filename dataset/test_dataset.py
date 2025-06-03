import torch
from dataset.dataset_gnn_dyn import ParticleDataset
from utils import load_yaml

def test_dataset():
    config = load_yaml('config/train/gnn_dyn.yaml')
        
    try:
        dataset = ParticleDataset(config['train']['data_root'], config, 'train')
        print(f"Dataset length: {len(dataset)}")
        
        # Test loading one sample
        states, states_delta, attrs, particle_num = dataset[0]
        print(f"Sample shapes:")
        print(f"  states: {states.shape}")
        print(f"  states_delta: {states_delta.shape}")
        print(f"  attrs: {attrs.shape}")
        print(f"  particle_num: {particle_num}")
        print("✅ Dataset test successful!")
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()