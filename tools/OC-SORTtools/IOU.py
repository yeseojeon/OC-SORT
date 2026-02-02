def Intersection(observation,prediction):
    observation_right = observation[0]+observation[2]/2
    observation_bottom = observation[1]+observation[3]/2
    observation_left = observation[0]-observation[2]/2
    observation_top = observation[1]-observation[3]/2

    #print(observation_right)


    prediction_right = prediction[0]+prediction[2]/2
    prediction_bottom = prediction[1]+prediction[3]/2
    prediction_left = prediction[0]-prediction[2]/2
    prediction_top = prediction[1]-prediction[3]/2
    
    #print(prediction_right)


    intersection_right = min(observation_right,prediction_right)
    intersection_bottom = min(observation_bottom,prediction_bottom)
    intersection_left = max(observation_left,prediction_left)
    intersection_top = max(observation_bottom,prediction_bottom)

  


    width = intersection_right-intersection_left
    height = intersection_bottom-intersection_top
    #print("\n")
    #print(width)
    #print(height)
    #print("\n")
    area = width*height
    if (area>0):
        return width*height
    else:
        return 0



def IOU(observation,prediction):
    observation_area = observation[2]*observation[3]
    prediction_area = prediction[2]*prediction[3]
    intersection = Intersection(observation,prediction)
    union = observation_area+prediction_area-intersection
    return intersection/union


  