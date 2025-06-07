import h5py
import argparse
from scripts.utils import parse_episodes
import random

def merge_datasets(file1, file2, episode_list1, episode_list2, output_file):
    count = 0
    with h5py.File(file1, 'r') as f1:
        with h5py.File(file2, 'r') as f2:
            with h5py.File(output_file, 'w') as fout: # Open the output file in write mode
                # Verify that the episodes exist in the files
                for episode_id in episode_list1:
                    episode_name = f'episode_{episode_id:06d}'
                    if episode_name not in f1:
                        print(f"Warning: {episode_name} not found in {file1}")
                        episode_list1.remove(episode_id)
                for episode_id in episode_list2:
                    episode_name = f'episode_{episode_id:06d}'
                    if episode_name not in f2:
                        print(f"Warning: {episode_name} not found in {file2}")
                        episode_list2.remove(episode_id)

                # Shuffle the indices
                indices = list(range(len(episode_list1)+len(episode_list2)))
                random.shuffle(indices)
                
                # Merge the episodes
                for episode_id in episode_list1:
                    episode_name = f'episode_{episode_id:06d}'
                    new_episode_name = f'episode_{indices[count]:06d}'
                    f1.copy(f1[episode_name], fout, name=new_episode_name)
                    count += 1

                for episode_id in episode_list2:
                    episode_name = f'episode_{episode_id:06d}'
                    new_episode_name = f'episode_{indices[count]:06d}'
                    f2.copy(f2[episode_name], fout, name=new_episode_name)
                    count += 1

    print(f"Merged {count} episodes into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge specified episodes from two H5 files into a new H5 file.")
    parser.add_argument('--data_dir', type=str, default='PhysTwin/generated_data/', help='Path to the data directory.')
    parser.add_argument('--file1', type=str, help='Name of the first H5 file.')
    parser.add_argument('--file2', type=str, help='Name of the second H5 file.')
    parser.add_argument('--output_file', type=str, help='Name of the output H5 file.')
    parser.add_argument('--episodes1', nargs='+', type=str, required=True,
                        help='Episode ranges to merge from file1 (e.g., "1-5,7").')
    parser.add_argument('--episodes2', nargs='+', type=str, required=True,
                        help='Episode ranges to merge from file2 (e.g., "10-12,15").')

    args = parser.parse_args()

    episode_list1 = parse_episodes(args.episodes1)
    episode_list2 = parse_episodes(args.episodes2)

    merge_datasets(args.data_dir + args.file1, args.data_dir + args.file2, episode_list1, episode_list2, args.data_dir + args.output_file)