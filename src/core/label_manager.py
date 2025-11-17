"""DeepLabCut labeling and keypoint management"""
from pathlib import Path
from typing import Optional
import deeplabcut
import yaml


class LabelManager:
    """Handles frame labeling, keypoint CRUD, and skeleton building"""
    
    def label_frames(self, config: str) -> None:
        """
        Launch DeepLabCut labeling GUI
        
        Args:
            config: Path to config.yaml
        """
        deeplabcut.label_frames(config)
    
    def get_bodyparts(self, config: str) -> list[str]:
        """Get list of bodyparts from config"""
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('bodyparts', [])
    
    def add_bodypart(self, config: str, bodypart: str) -> None:
        """Add a new bodypart to config"""
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        bodyparts = cfg.get('bodyparts', [])
        if bodypart not in bodyparts:
            bodyparts.append(bodypart)
            cfg['bodyparts'] = bodyparts
            
            with open(config, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False)
    
    def remove_bodypart(self, config: str, bodypart: str) -> None:
        """Remove a bodypart from config"""
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        bodyparts = cfg.get('bodyparts', [])
        if bodypart in bodyparts:
            bodyparts.remove(bodypart)
            cfg['bodyparts'] = bodyparts
            
            with open(config, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False)
    
    def update_bodypart(self, config: str, old_name: str, new_name: str) -> None:
        """Update bodypart name"""
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        bodyparts = cfg.get('bodyparts', [])
        if old_name in bodyparts:
            idx = bodyparts.index(old_name)
            bodyparts[idx] = new_name
            cfg['bodyparts'] = bodyparts
            
            with open(config, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False)
    
    def get_skeleton(self, config: str) -> list[list[str]]:
        """Get skeleton connections from config"""
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('skeleton', [])
    
    def add_skeleton_connection(self, config: str, bp1: str, bp2: str) -> None:
        """Add skeleton connection between two bodyparts"""
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        skeleton = cfg.get('skeleton', [])
        connection = [bp1, bp2]
        
        if connection not in skeleton and [bp2, bp1] not in skeleton:
            skeleton.append(connection)
            cfg['skeleton'] = skeleton
            
            with open(config, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False)
    
    def remove_skeleton_connection(self, config: str, bp1: str, bp2: str) -> None:
        """Remove skeleton connection"""
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        skeleton = cfg.get('skeleton', [])
        connection = [bp1, bp2]
        reverse = [bp2, bp1]
        
        if connection in skeleton:
            skeleton.remove(connection)
        elif reverse in skeleton:
            skeleton.remove(reverse)
        
        cfg['skeleton'] = skeleton
        
        with open(config, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
    
    def check_labels(self, config: str) -> dict:
        """Check labeling status"""
        try:
            with open(config, 'r') as f:
                cfg = yaml.safe_load(f)
            
            project_path = Path(config).parent
            labeled_data_path = project_path / 'labeled-data'
            
            if not labeled_data_path.exists():
                return {'status': 'No labeled-data folder found'}
            
            # Count labeled frames
            total_videos = 0
            total_frames = 0
            labeled_frames = 0
            
            for video_dir in labeled_data_path.iterdir():
                if video_dir.is_dir():
                    total_videos += 1
                    # Check for h5 files (labeled data)
                    h5_files = list(video_dir.glob('CollectedData_*.h5'))
                    csv_files = list(video_dir.glob('CollectedData_*.csv'))
                    
                    if h5_files or csv_files:
                        # Count frames in this video folder
                        frames = list(video_dir.glob('img*.png'))
                        total_frames += len(frames)
                        
                        # If h5 exists, count labeled frames
                        if h5_files:
                            import pandas as pd
                            df = pd.read_hdf(h5_files[0])
                            labeled_frames += len(df)
            
            if total_videos == 0:
                return {'status': 'No video folders found in labeled-data'}
            
            return {
                'Videos': total_videos,
                'Total Frames': total_frames,
                'Labeled Frames': labeled_frames,
                'Completion': f'{labeled_frames}/{total_frames}' if total_frames > 0 else 'N/A'
            }
            
        except Exception as e:
            return {'error': str(e)}
