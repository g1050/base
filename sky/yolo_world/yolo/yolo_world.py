from ultralytics import YOLO
 
 
if __name__ == '__main__':
 
    # Initialize a YOLO-World model
    model = YOLO('download/yolov8s-world.pt')  # or choose yolov8m/l-world.pt
 
    # Define custom classes
    model.set_classes(["Rottweiler"])
 
    # Execute prediction for specified categories on an image
    results = model.predict('img/Rottweiler.jpg')
 
    # Show results
    results[0].show()