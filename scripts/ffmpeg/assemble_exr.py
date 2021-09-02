import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", default="./raw_data/sph3d",
                    help="render folder")
parser.add_argument("--out_path", default="./render_data/sph_waterdrop.mp4",
                    help="output video file")
parser.add_argument("--fps", default="20",
                    help="video fps")
parser.add_argument("--resolution", default="scale=768:-1",
                    help="video resolution")

args = parser.parse_args()

def combine_video(_in,_out,resolution,fps=60):
    out_video_path = _out
    out_args = [
        'ffmpeg',
        '-gamma', '2.2',
        '-i', '%s/%%04d.exr' % _in,
        '-f', 'mp4',
        '-q:v', '0',
        '-vcodec', 'mpeg4',
        '-r', str(fps),
		'-vf', str(resolution),
        out_video_path
    ]
    
    subprocess.call(" ".join(out_args), shell=True)
    print('output video at: %s' % out_video_path)

def main():
	combine_video(args.in_path,args.out_path,args.resolution,args.fps)

if __name__ == "__main__":
    main()