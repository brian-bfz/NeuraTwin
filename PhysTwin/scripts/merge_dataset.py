import h5py
import argparse
from scripts.utils import parse_episodes
import random

def merge_datasets(file1, file2, output_file):
    count = 0
    with h5py.File(file1, 'r') as f1:
        with h5py.File(file2, 'r') as f2:
            with h5py.File(output_file, 'w') as fout: # Open the output file in write mode
                # Discover all episodes in both files
                episode_list1 = []
                episode_list2 = []
                
                # Get all episode keys from file1
                for key in f1.keys():
                    if key.startswith('episode_'):
                        episode_id = int(key.split('_')[1])
                        episode_list1.append(episode_id)
                
                # Get all episode keys from file2
                for key in f2.keys():
                    if key.startswith('episode_'):
                        episode_id = int(key.split('_')[1])
                        episode_list2.append(episode_id)
                
                print(f"Found {len(episode_list1)} episodes in {file1}")
                print(f"Found {len(episode_list2)} episodes in {file2}")

                # Shuffle the indices
                indices = list(range(len(episode_list1)+len(episode_list2)))
                random.shuffle(indices)
                print(indices)
                
                # Merge the episodes from file1
                for episode_id in episode_list1:
                    episode_name = f'episode_{episode_id:06d}'
                    new_episode_name = f'episode_{indices[count]:06d}'
                    f1.copy(f1[episode_name], fout, name=new_episode_name)
                    count += 1

                # Merge the episodes from file2
                for episode_id in episode_list2:
                    episode_name = f'episode_{episode_id:06d}'
                    new_episode_name = f'episode_{indices[count]:06d}'
                    f2.copy(f2[episode_name], fout, name=new_episode_name)
                    count += 1

    print(f"Merged {count} episodes into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all episodes from two H5 files into a new H5 file.")
    parser.add_argument('--data_dir', type=str, default='PhysTwin/generated_data/', help='Path to the data directory.')
    parser.add_argument('--file1', type=str, help='Name of the first H5 file.')
    parser.add_argument('--file2', type=str, help='Name of the second H5 file.')
    parser.add_argument('--output_file', type=str, help='Name of the output H5 file.')

    args = parser.parse_args()

    merge_datasets(args.data_dir + args.file1, args.data_dir + args.file2, args.data_dir + args.output_file)