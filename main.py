import cv2
import argparse
from tracker import LinearTracker
from cases import blur_Image, sharpen_Image, pass_Image

def main(args):
    mode = args.blur
    blur = lambda frame : blur_Image(51, 10, frame) if mode==1 else sharpen_Image(30, 12, frame) if mode==2 else pass_Image(frame)
    # Open video file                
    cap = cv2.VideoCapture(args.filename)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    # Select initial ROI
    roi = cv2.selectROI("Select target", frame, False)
    cv2.destroyWindow("Select target")

    tracker = LinearTracker(args.sigma, args.lambda_val, args.interp, mode)
    tracker.init(frame, roi)
    
    pause = False
    while True:
        if not pause:
            ret, frame = cap.read()
            frame = blur(frame)
            if not ret:
                break
        
        # Make a copy of the frame for drawing
        display_frame = frame.copy()
            
        # Update tracker
        roi, z = tracker.update(frame)
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27: # q or esc to exit
            break

        elif key == ord('f'):
            # select new ROI
            roi = cv2.selectROI("Select new target", display_frame, False)
            cv2.destroyWindow("Select new target")
            
            # Reinitialize tracker with new ROI
            tracker.init(frame, roi)

        elif key == ord(' '):  # Space bar to pause/unpause
            pause = not pause

        # Draw bounding box
        if roi is not None:
            x, y, w, h = map(int, roi)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if args.show_roi and z.shape[0] > 0 and z.shape[1] > 0:
                cv2.imshow("ROI", cv2.resize(z, (w,h)))
        
        # Display frame
        cv2.imshow("Tracking", display_frame)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help="Name of mp4 file to run")
    parser.add_argument('-s', '--sigma', type=float, help="sigma value for gaussian target")
    parser.add_argument('-l', '--lambda_val', type=float, help="lambda value for filter")
    parser.add_argument('-iv', '--interp', type=float, help="interpolation value for tracker")
    parser.add_argument('-roi', '--show_roi', action='store_true', help="Display raw ROI on separate screen")
    parser.add_argument('-b', '--blur', type=int, default=0, help="blur to add to video. Specify 1 or 2.")
    main(parser.parse_args())