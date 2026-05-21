import argparse

from RawForge.application.convert import convert_image

def main():
    parser = argparse.ArgumentParser(description='Image converter')
    parser.add_argument('--in_file', type=str, help='Source file')
    parser.add_argument('--out_file', type=str, help='Target file')
    args = parser.parse_args()
    
    if not args.in_file:
        print('--in_file required')
        return
    
    if not args.out_file:
        print('--out_file required')
        return

    convert_image(args.in_file, args.out_file)

if __name__ == '__main__':
    main()
