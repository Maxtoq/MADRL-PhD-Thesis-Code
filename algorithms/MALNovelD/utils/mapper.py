import copy

class Mapper:

    def __init__(self, sce_conf):
        """
        Mapper, set maps to track the agents and get what he sees
        :param sce_conf: (dict) parameters of the exercise
        """
        self.sce_conf = sce_conf

        # Initialization of the world map
        # To track the agent
        self.world = []
        for nb_agent in range(self.sce_conf['nb_agents']):
            newAgent = []
            for line in range(6):
                newLine = []
                for col in range(6):
                    newLine.append(0)
                newAgent.append(newLine)
            self.world.append(newAgent)
        

        # Initialization of the area map
        # Each 0 is an area (North, South, West, East)
        # That has not been fully discovered
        self.area = []
        for nb_agent in range(self.sce_conf['nb_agents']):
            newAgent = []
            for line in range(3):
                newLine = []
                for col in range(3):
                    newLine.append(0)
                newAgent.append(newLine)
            self.area.append(newAgent)

        # Initialization of the area map with objects
        # Each 0 is the number of objects found in the area
        self.area_obj = []
        for nb_agent in range(self.sce_conf['nb_agents']):
            newAgent = []
            for line in range(3):
                newLine = []
                for col in range(3):
                    newLine.append(0)
                newAgent.append(newLine)
            self.area_obj.append(newAgent)

    # Update the value of the world map
    def update_world(self, posX, posY, nb_agent):
        '''
        Update the world array depending on the position of the agent

        Input:  
            posX: (float) between -1 and 1, position of the agent
            posY: (float) between -1 and 1, position of the agent
            nb_agent: (int) Number of the agent
        '''

        #Check the position of the agent
        # To update the value of the world
        
        # North
        # Can be in a corner
        if posY >= 0.66 :
            self.update_world_section(nb_agent,0,posX,corner = True)
        elif posY >= 0.33 and posY <= 0.66 :
            self.update_world_section(nb_agent,1,posX,corner = True)

        # Center
        if posY >= 0 and posY <= 0.33 :
            self.update_world_section(nb_agent,2,posX,corner = False)
        elif posY >= -0.33 and posY <= 0 :
            self.update_world_section(nb_agent,3,posX,corner = False)

        # South 
        # Can be in a corner
        if posY >= -0.66 and posY <= -0.33 :
            self.update_world_section(nb_agent,4,posX,corner = True)
        elif posY <= -0.66:
            self.update_world_section(nb_agent,5,posX,corner = True)

    # Update a section of the world map
    def update_world_section(self, nb_agent, num_array, posX, corner):
        '''
        Update a section of the world array depending on the position of the agent
        The areas not discovered =0, =1 when discovered and =2 when discovered and
        in a corner

        Input: 
            nb_agent: (int) Number of the agent 
            num_array: (int) number of the array we need to modify
            posX: (float) between -1 and 1, position of the agent
            corner: (bool) if True the agent is in a corner
        '''
        # 0 means not discovered
        # 1 means discovered
        # 2 means discovered in corners

        if corner == True:
            # In a corner so = 2 (for the two possible areas)
            if posX <= -0.66:
                self.world[nb_agent][num_array][0] = 2
            elif posX >= -0.66 and posX <= -0.33:
                self.world[nb_agent][num_array][1] = 2
            elif posX >= -0.33 and posX <= 0:
                self.world[nb_agent][num_array][2] = 1
            elif posX >= 0 and posX <= 0.33:
                self.world[nb_agent][num_array][3] = 1
            elif posX >= 0.33 and posX <= 0.66:
                self.world[nb_agent][num_array][4] = 2
            elif posX >= 0.66:
                self.world[nb_agent][num_array][5] = 2
        else:
            if posX <= -0.66:
                self.world[nb_agent][num_array][0] = 1
            elif posX >= -0.66 and posX <= -0.33:
                self.world[nb_agent][num_array][1] = 1
            elif posX >= -0.33 and posX <= 0:
                self.world[nb_agent][num_array][2] = 1
            elif posX >= 0 and posX <= 0.33:
                self.world[nb_agent][num_array][3] = 1
            elif posX >= 0.33 and posX <= 0.66:
                self.world[nb_agent][num_array][4] = 1
            elif posX >= 0.66:
                self.world[nb_agent][num_array][5] = 1

    # Update the array of objects
    def update_area_obj(self, agent_x, agent_y, object_nb, nb_agent):
        '''
        Update the object array of the agent depending on what it sees and where
        
        object_nb : 2 if object
                    3 if landmark
                    5 if both

        Input: 
            agent_x: (float) Position x of the agent 
            agent_y: (float) Position y of the agent
            object_nb: (int) each object has a different number
            nb_agent: (int) number of the agent
        '''
        
        # If an object is discovered, we modify the area_obj array
        
        # North
        if agent_y >= 0.33:
            if agent_x >= 0.33:
                # If the object is different than
                # An object seen in the same area
                if (self.area_obj[nb_agent][0][2] != 0 and 
                self.area_obj[nb_agent][0][2] != object_nb and 
                self.area_obj[nb_agent][0][2] != object_nb*2):
                    self.area_obj[nb_agent][0][2] = 5
                # Else, we are in a corner
                # So we multiply the object num by 2
                else :
                    self.area_obj[nb_agent][0][2] = object_nb*2
            elif agent_x <= -0.33:
                if (self.area_obj[nb_agent][0][0] != 0 and 
                self.area_obj[nb_agent][0][0] != object_nb and 
                self.area_obj[nb_agent][0][0] != object_nb*2):
                    self.area_obj[nb_agent][0][0] = 5
                else :
                    self.area_obj[nb_agent][0][0] = object_nb*2
            else :
                if (self.area_obj[nb_agent][0][1] != 0 and 
                self.area_obj[nb_agent][0][1] != object_nb and 
                self.area_obj[nb_agent][0][1] != object_nb*2):
                    self.area_obj[nb_agent][0][1] = 5
                else :
                    self.area_obj[nb_agent][0][1] = object_nb
        # South
        elif agent_y <= -0.33:
            if agent_x >= 0.33:
                if (self.area_obj[nb_agent][2][2] != 0 and 
                self.area_obj[nb_agent][2][2] != object_nb and 
                self.area_obj[nb_agent][2][2] != object_nb*2):
                    self.area_obj[nb_agent][2][2] = 5
                else :
                    self.area_obj[nb_agent][2][2] = object_nb*2
            elif agent_x <= -0.33:
                if (self.area_obj[nb_agent][2][0] != 0 and 
                self.area_obj[nb_agent][2][0] != object_nb and 
                self.area_obj[nb_agent][2][0] != object_nb*2):
                    self.area_obj[nb_agent][2][0] = 5
                else :
                    self.area_obj[nb_agent][2][0] = object_nb*2
            else :
                if (self.area_obj[nb_agent][2][1] != 0 and 
                self.area_obj[nb_agent][2][1] != object_nb and 
                self.area_obj[nb_agent][2][1] != object_nb*2):
                    self.area_obj[nb_agent][2][1] = 5
                else :
                    self.area_obj[nb_agent][2][1] = object_nb
        
        # Center
        else:
            if agent_x >= 0.33:
                if (self.area_obj[nb_agent][1][2] != 0 and 
                self.area_obj[nb_agent][1][2] != object_nb and 
                self.area_obj[nb_agent][1][2] != object_nb*2):
                    self.area_obj[nb_agent][1][2] = 5
                else :
                    self.area_obj[nb_agent][1][2] = object_nb
            elif agent_x <= -0.33:
                if (self.area_obj[nb_agent][1][0] != 0 and 
                self.area_obj[nb_agent][1][0] != object_nb and 
                self.area_obj[nb_agent][1][0] != object_nb*2):
                    self.area_obj[nb_agent][1][0] = 5
                else :
                    self.area_obj[nb_agent][1][0] = object_nb
            else :
                if (self.area_obj[nb_agent][1][1] != 0 and 
                self.area_obj[nb_agent][1][1] != object_nb and 
                self.area_obj[nb_agent][1][1] != object_nb*2):
                    self.area_obj[nb_agent][1][1] = 5
                else :
                    self.area_obj[nb_agent][1][1] = object_nb

    # Check if an area is fully dicovered
    def count_discovered(self, nb_agent, world_array_posx, world_array_posy):
        '''
        Check an area to see if it has been fully discovered
        by counting the number of cells the agent went through

        Input: 
            nb_agent: (int) number of the agent
            world_array_posx:   list(int) list of the different array 
                                number we have to check 
            world_array_posy:   list(int) list of the different array 
                                number we have to check

        Output:
            (bool) True if the area is fully discovered
        '''
        max_count = 0
        count = 0

        # Count the number of cell discovered
        for i in range(world_array_posx[0], world_array_posx[1]):
            for j in range(world_array_posy[0], world_array_posy[1]):
                max_count += 1
                # if == 1 or == 2, the agent saw it
                if self.world[nb_agent][i][j] >= 1 :
                    count += 1
                else:
                    break
        
        # Returns True or False depending on 
        # If the agent saw it entirely
        if count == max_count:
            return True
        else:
            return False

    # Check in a corner area is fully discovered
    def count_discovered_corner(self, nb_agent, world_array_posx, world_array_posy):
        '''
        Check an area in a corner to see if it has been fully discovered
        by counting the number of cells the agent went through

        Input: 
            nb_agent: (int) number of the agent
            world_array_posx:   list(int) list of the different array 
                                number we have to check 
            world_array_posy:   list(int) list of the different array 
                                number we have to check

        Output:
            (bool) True if the area is fully discovered
        '''
        max_count = 0
        count = 0

        # Count the number of cell discovered (corner)
        for i in range(world_array_posx[0], world_array_posx[1]):
            for j in range(world_array_posy[0], world_array_posy[1]):
                max_count += 1
                # if == 2, the agent saw the corner
                if self.world[nb_agent][i][j] > 1 :
                    count += 1
                else:
                    break
        
        # Returns True or False depending on 
        # If the agent saw it entirely
        if count == max_count:
            return True
        else:
            return False

    # Update the area map
    def update_area(self, nb_agent):     
        '''
        Update the area array of the agent depending on what it sees
        If an area is fully discovered we need to generate a not_sentence
        If it's the case, we return the informations on the array 
        (its position and the number of the agent)
        If the area is in a corner we set its value to 2 instead of 1 
        so we can generate not_sentence twice for the corner
        
        Input: 
            nb_agent: (int) number of the agent
                    
        Output:
            list(
                (int) Number of the area
                (int) Number of the area
                (int) Number of the agent
                )
        '''
        # Check the world to see if some area were fully discovered
        # If North is not fully discovered
        if (self.area[nb_agent][0][0] < 1 or 
            self.area[nb_agent][0][1] < 1 or 
            self.area[nb_agent][0][2] < 1):
            # North West 
            if self.area[nb_agent][0][0] < 2 :
                if self.count_discovered_corner(nb_agent,[0,2],[0,2]):
                        self.area[nb_agent][0][0] = 2
                        # Generate a not sentence
                        return 0, 0, nb_agent
            # North Center
            if self.area[nb_agent][0][1] != 1:
                if self.count_discovered(nb_agent,[0,2],[2,4]):
                    if self.area[nb_agent][0][1] == 0:
                        self.area[nb_agent][0][1] = 1
            # North East
            if self.area[nb_agent][0][2] < 2:
                if self.count_discovered_corner(nb_agent,[0,2],[4,6]):
                        self.area[nb_agent][0][2] = 2
                        # Generate a not sentence
                        return 0, 2, nb_agent
        else:
            return 0, 1, nb_agent

        # If Center not fully discovered
        if (self.area[nb_agent][1][0] != 1 or 
            self.area[nb_agent][1][1] != 1 or
            self.area[nb_agent][1][2] != 1):
            # Center West 
            if self.area[nb_agent][1][0] != 1:
                if self.count_discovered(nb_agent,[2,4],[0,2]):
                    if self.area[nb_agent][1][0] == 0:
                        self.area[nb_agent][1][0] = 1
            # Center Center
            if self.area[nb_agent][1][1] != 1:
                if self.count_discovered(nb_agent,[2,4],[2,4]):
                    if self.area[nb_agent][1][1] == 0:
                        self.area[nb_agent][1][1] = 1
                    return 1, 1, nb_agent
            # Center East
            if self.area[nb_agent][1][2] != 1:
                if self.count_discovered(nb_agent,[2,4],[4,6]):
                    if self.area[nb_agent][1][2] == 0:
                        self.area[nb_agent][1][2] = 1

        # If South is not fully discovered
        if (self.area[nb_agent][2][0] < 1 or 
            self.area[nb_agent][2][1] < 1 or 
            self.area[nb_agent][2][2] < 1):
            # South West 
            if self.area[nb_agent][2][0] < 2:
                if self.count_discovered_corner(nb_agent,[4,6],[0,2]):
                        self.area[nb_agent][2][0] = 2
                        return 2, 0, nb_agent
            # South Center
            if self.area[nb_agent][2][1] != 1:
                if self.count_discovered(nb_agent,[4,6],[2,4]):
                    if self.area[nb_agent][2][1] == 0:
                        self.area[nb_agent][2][1] = 1
            # South East
            if self.area[nb_agent][2][2] < 2:
                if self.count_discovered_corner(nb_agent,[4,6],[4,6]):
                        self.area[nb_agent][2][2] = 2
                        return 2, 2, nb_agent
        else:
            return 2, 1, nb_agent

        # West and East
        # If the 3 areas were discovered
        if (self.area[nb_agent][0][0] >= 1 and 
            self.area[nb_agent][1][0] >= 1 and
            self.area[nb_agent][2][0] >= 1):
            # Generate the not_sentence
            return 1, 0, nb_agent
        if (self.area[nb_agent][0][2] >= 1 and 
            self.area[nb_agent][1][2] >= 1 and
            self.area[nb_agent][2][2] >= 1):
            return 1, 2, nb_agent

    # Reset the area map
    def reset_area(self, nb_agent, direction, area_pos):
        '''
        Reset the area and set its value to 0 (or 1 if it was a corner == 2)
        Also reset the object array by setting its value to 2 or diving it by 2
        if it was a corner
        
        Input: 
            nb_agent: (int) number of the agent
            direction:  (int) number of the direction. 
                        0 is horizontal
                        1 if vertical 
            area_pos: (int) position of the area in the array
        '''
        # If direction is left to right (North or South)
        if direction == 0:
            for i in range(3):
                # Reset the area by doing -1
                self.area[nb_agent][area_pos][i] -= 1
                # If the agent saw an object (and in a corner)
                if (self.area_obj[nb_agent][area_pos][i] >= 4 and \
                    self.area_obj[nb_agent][area_pos][i] != 5):
                    # Devide it by 2
                    self.area_obj[nb_agent][area_pos][i] = \
                        self.area_obj[nb_agent][area_pos][i]//2
                elif self.area[nb_agent][area_pos][i] == 0:
                    # Or = 0
                    self.area_obj[nb_agent][area_pos][i] = 0

        # If direction is up to bottom (West or East)
        elif direction == 1:
            for i in range(3):
                self.area[nb_agent][i][area_pos] -= 1
                if (self.area_obj[nb_agent][i][area_pos] >= 4 and \
                    self.area_obj[nb_agent][i][area_pos] != 5):
                    self.area_obj[nb_agent][i][area_pos] = \
                        self.area_obj[nb_agent][i][area_pos]//2
                elif self.area[nb_agent][i][area_pos] == 0:
                    self.area_obj[nb_agent][i][area_pos] = 0

    # Reset the world map
    def reset_world(self, nb_agent, world_array_posx, world_array_posy):
        '''
        Reset the world array and set all its value to 0
        
        Input: 
            nb_agent: (int) number of the agent
            world_array_posx:   list(int) list of the different array 
                                number we have to check 
            world_array_posy:   list(int) list of the different array 
                                number we have to check
        '''
        # Reset the world map depending on the array position
        for i in range(world_array_posx[0],world_array_posx[1]) :
                for j in range(world_array_posy[0],world_array_posy[1]):
                    self.world[nb_agent][i][j] -= 1 

    # Reset all the maps
    def reset_areas(self, area_nb, nb_agent):
        '''
        Reset the values of the areas
        
        Input: 
            area_nb: (int) number of the area to reset
            nb_agent: (int) number of the agent
        '''

        # If North
        if area_nb == 0:
            # Reset the area and the world map
            # Send area values to other programs
            self.reset_area(nb_agent,0,0)
            self.reset_world(nb_agent,[0,2],[0,6])
        # If South
        elif area_nb == 1:
            self.reset_area(nb_agent,0,2)
            self.reset_world(nb_agent,[4,6],[0,6])
        # if West
        elif area_nb == 2:
            self.reset_area(nb_agent,1,0)
            self.reset_world(nb_agent,[0,6],[0,2])
        # If East
        elif area_nb == 3:
            self.reset_area(nb_agent,1,2)
            self.reset_world(nb_agent,[0,6],[4,6])
        
        # If Center
        elif area_nb == 4:
            # Reset the area (1,1)
            self.area[nb_agent][1][1] -= 1
            if self.area_obj[nb_agent][1][1] >= 4:
                self.area_obj[nb_agent][1][1] = \
                    self.area_obj[nb_agent][1][1]//2
            else:
                self.area_obj[nb_agent][1][1] = 0
            # Reset the world
            self.reset_world(nb_agent,[2,4],[2,4])

    # Check an area to see if there is an object
    def check_area(self, nb_agent, direction, area_pos, obj):
        '''
        Check an selected area to see if there is 1 type of object or 2
        if obj == 2: there is an object
        if obj == 3: there is a landmark
        if obj == 5: both objects are in the same area
        
        Input: 
            nb_agent: (int) number of the agent
            direction: (int) direction of the area 
            area_pos: (int) position of the area 
            obj: (int) value describing the type of object in the area

        Output:
            obj: (int) return the value describing the type of object in the area
        '''
        # If North or South
        if direction == 0:
            for x in range(3) :
                    # obj_i represente the object found in the area
                    obj_i = self.area_obj[nb_agent][area_pos][x]
                    # If 2 or 4 then obj = 2
                    if ((obj_i == 2 or obj_i == 4) and 
                    (obj == 2 or obj == 4 or obj == 0)):
                        obj = 2
                    # if 3 or 6 then obj = 3
                    elif ((obj_i == 3 or obj_i == 6) and 
                    (obj == 3 or obj == 6 or obj == 0)):
                        obj = 3
                    elif obj_i != 0 :
                        # Else, it means that two different objects
                        # Are in the same area, obj = 4
                        obj = 5
        # If West or East
        elif direction == 1:
            for x in range(3) :
                    obj_i = self.area_obj[nb_agent][x][area_pos]
                    if ((obj_i == 2 or obj_i == 4) and 
                    (obj == 2 or obj == 4 or obj == 0)):
                        obj = 2
                    elif ((obj_i == 3 or obj_i == 6) and 
                    (obj == 3 or obj == 6 or obj == 0)):
                        obj = 3
                    elif obj_i != 0 :
                        obj = 5

        return obj

    # Reset all the maps
    def reset(self):
        '''
        Reset all the arrays
        '''
        # Initialization of the world map
        # To track the agent
        self.world = []
        for nb_agent in range(self.sce_conf['nb_agents']):
            newAgent = []
            for line in range(6):
                newLine = []
                for col in range(6):
                    newLine.append(0)
                newAgent.append(newLine)
            self.world.append(newAgent)
        

        # Initialization of the area map
        # Each 0 is an area (North, South, West, East)
        # That has not been fully discovered
        self.area = []
        for nb_agent in range(self.sce_conf['nb_agents']):
            newAgent = []
            for line in range(3):
                newLine = []
                for col in range(3):
                    newLine.append(0)
                newAgent.append(newLine)
            self.area.append(newAgent)

        # Initialization of the area map with objects
        # Each 0 is the number of objects found in the area
        self.area_obj = []
        for nb_agent in range(self.sce_conf['nb_agents']):
            newAgent = []
            for line in range(3):
                newLine = []
                for col in range(3):
                    newLine.append(0)
                newAgent.append(newLine)
            self.area_obj.append(newAgent)


class ColorMapper(Mapper):

    def __init__(self, sce_conf):
        """
        Mapper, set maps to track the agents and see what he sees
        :param sce_conf: (dict) parameters of the exercise
        """
        self.sce_conf = sce_conf

        # Initialization of the world map
        # To track the agent
        self.world = []
        for nb_agent in range(self.sce_conf['nb_agents']):
            newAgent = []
            for line in range(6):
                newLine = []
                for col in range(6):
                    newLine.append(0)
                newAgent.append(newLine)
            self.world.append(newAgent)
        

        # Initialization of the area map
        # Each 0 is an area (North, South, West, East)
        # That has not been fully discovered
        self.area = []
        for nb_agent in range(self.sce_conf['nb_agents']):
            newAgent = []
            for line in range(3):
                newLine = []
                for col in range(3):
                    newLine.append(0)
                newAgent.append(newLine)
            self.area.append(newAgent)

        self.area_object = []
        for nb_agent in range(self.sce_conf['nb_agents']):
            newAgent = []
            for line in range(3):
                newLine = []
                for col in range(3):
                    obj = [0]
                    newLine.append(obj)
                newAgent.append(newLine)
            self.area_object.append(newAgent)

    # Update the array of objects
    def update_area_obj(self, agent_x, agent_y, object_nb, object_color, nb_agent):
        '''
        Update the object array of the agent depending on what it sees and where
        
        object_nb : 2 if object
                    3 if landmark

        Input: 
            agent_x: (float) Position x of the agent 
            agent_y: (float) Position y of the agent
            object_nb: (int) each object has a different number
            object_color: (int) color of the object
            nb_agent: (int) number of the agent
        '''
        # If an object is discovered, we modify the area_obj array

        object = [object_nb,object_color]
        # North
        if agent_y >= 0.33:
            if agent_x >= 0.33:
                if object not in self.area_object[nb_agent][0][2]:
                    self.area_object[nb_agent][0][2].append(object)
                    # Corner so = 2
                if self.area_object[nb_agent][0][2][0] < 2:
                    self.area_object[nb_agent][0][2][0] = 2
            elif agent_x <= -0.33:
                if object not in self.area_object[nb_agent][0][0]:
                    self.area_object[nb_agent][0][0].append(object)
                    # Corner so = 2
                if self.area_object[nb_agent][0][0][0] < 2:
                    self.area_object[nb_agent][0][0][0] = 2
            else :
                if object not in self.area_object[nb_agent][0][1]:
                    self.area_object[nb_agent][0][1].append(object)
                    # Not corner so = 1
                if self.area_object[nb_agent][0][1][0] == 0:
                    self.area_object[nb_agent][0][1][0] = 1
        # South
        elif agent_y <= -0.33:
            if agent_x >= 0.33:
                if object not in self.area_object[nb_agent][2][2]:
                    self.area_object[nb_agent][2][2].append(object)
                    # Corner so = 2
                if self.area_object[nb_agent][2][2][0] < 2:
                    self.area_object[nb_agent][2][2][0] = 2
            elif agent_x <= -0.33:
                if object not in self.area_object[nb_agent][2][0]:
                    self.area_object[nb_agent][2][0].append(object)
                # Corner so = 2
                if self.area_object[nb_agent][2][0][0] < 2:
                    self.area_object[nb_agent][2][0][0] = 2
            else :
                if object not in self.area_object[nb_agent][2][1]:
                    self.area_object[nb_agent][2][1].append(object)
                # Not corner so = 1
                if self.area_object[nb_agent][2][1][0] == 0:
                    self.area_object[nb_agent][2][1][0] = 1
        
        # Center
        else:
            if agent_x >= 0.33:
                if object not in self.area_object[nb_agent][1][2]:
                    self.area_object[nb_agent][1][2].append(object)
                # Corner so = 2
                if self.area_object[nb_agent][1][2][0] == 0:
                    self.area_object[nb_agent][1][2][0] = 1
            elif agent_x <= -0.33:
                if object not in self.area_object[nb_agent][1][0]:
                    self.area_object[nb_agent][1][0].append(object)
                # Corner so = 2
                if self.area_object[nb_agent][1][0][0] == 0:
                    self.area_object[nb_agent][1][0][0] = 1
            else :
                if object not in self.area_object[nb_agent][1][1]:
                    self.area_object[nb_agent][1][1].append(object)
                # Not corner so = 1
                if self.area_object[nb_agent][1][1][0] == 0:
                    self.area_object[nb_agent][1][1][0] = 1

    # Reset the area map
    def reset_area(self, nb_agent, direction, area_pos):
        '''
        Reset the area and set its value to 0 (or 1 if it was a corner == 2)
        Also reset the object array by setting its value to 2 or diving it by 2
        if it was a corner
        
        Input: 
            nb_agent: (int) number of the agent
            direction:  (int) number of the direction. 
                        0 is horizontal
                        1 if vertical 
            area_pos: (int) position of the area in the array
        '''
        # If direction is left to right (North or South)
        if direction == 0:
            for i in range(3):
                # Reset the area by doing -1
                self.area[nb_agent][area_pos][i] -= 1

                # If the agent saw an object (and in a corner)
                if self.area_object[nb_agent][area_pos][i][0] == 2:
                    # Devide it by 2
                    self.area_object[nb_agent][area_pos][i][0] = 1
                else:
                    # Or = 0
                    self.area_object[nb_agent][area_pos][i] = [0]

        # If direction is up to bottom (West or East)
        elif direction == 1:
            for i in range(3):
                self.area[nb_agent][i][area_pos] -= 1

                if self.area_object[nb_agent][i][area_pos][0] == 2:
                    self.area_object[nb_agent][i][area_pos][0] = 1
                else:
                    self.area_object[nb_agent][i][area_pos] = [0]

    # Reset all the maps
    def reset_areas(self, area_nb, nb_agent):
        '''
        Reset the values of the areas
        
        Input: 
            area_nb: (int) number of the area to reset
            nb_agent: (int) number of the agent
        '''
        # If North
        if area_nb == 0:
            # Reset the area and the world map
            # Send area values to other programs
            self.reset_area(nb_agent,0,0)
            self.reset_world(nb_agent,[0,2],[0,6])
        # If South
        elif area_nb == 1:
            self.reset_area(nb_agent,0,2)
            self.reset_world(nb_agent,[4,6],[0,6])
        # if West
        elif area_nb == 2:
            self.reset_area(nb_agent,1,0)
            self.reset_world(nb_agent,[0,6],[0,2])
        # If East
        elif area_nb == 3:
            self.reset_area(nb_agent,1,2)
            self.reset_world(nb_agent,[0,6],[4,6])

        # If Center
        elif area_nb == 4:
            # Reset the area (1,1)
            self.area[nb_agent][1][1] -= 1
            if self.area_object[nb_agent][1][1][0] == 2:
                self.area_object[nb_agent][1][1][0] = 1
            else:
                self.area_object[nb_agent][1][1] = [0]
            # Reset the world
            self.reset_world(nb_agent,[2,4],[2,4])

    # Check an area to see if there is an object
    def check_area(self, nb_agent, direction, area_pos, all_colors):
        '''
        Check an selected area and return the list of object not in the area
        if obj == 2: there is an object
        if obj == 3: there is a landmark
        
        Input: 
            nb_agent: (int) number of the agent
            direction: (int) direction of the area 
            area_pos: (int) position of the area 
            all_colors: list(int) all the colors possible in the exercise

        Output:
            list_all_object: list(list(int)) return a list of object that are not in the area 
        '''
        objects = []
        list_all_obj = []
        for c in all_colors:
            if [2,c] not in list_all_obj and [3,c] not in list_all_obj:
                list_all_obj.append([2,c])
                list_all_obj.append([3,c])
        # If North or South
        if direction == 0:
            for x in range(3) :
                objects = self.area_object[nb_agent][area_pos][x]
                for obj in objects:
                    if obj in list_all_obj:
                        list_all_obj.remove(obj)

        # If West or East
        elif direction == 1:
            for x in range(3) :
                objects = self.area_object[nb_agent][x][area_pos]
                for obj in objects:
                    if obj in list_all_obj:
                        list_all_obj.remove(obj)

        return list_all_obj

    # Return list of object not visible
    def find_missing(self, nb_agent, posx, posy, all_colors):
        '''
        Check an selected area and return the list of object not in the area
        if obj == 2: there is an object
        if obj == 3: there is a landmark
        
        Input: 
            nb_agent: (int) number of the agent
            posx: (int) position in the array
            posy: (int) position in the array 
            all_colors: list(int) all the colors possible in the exercise

        Output:
            list_all_object: list(list(int)) return a list of object that are not in the area 
        '''
        list_obj = []
        list_all_obj = []
        objects_not_visible = []

        # Create a list with all the possible objects
        # Each color has an object and a landmark
        for c in all_colors:
            if [2,c] not in list_all_obj and [3,c] not in list_all_obj:
                list_all_obj.append([2,c])
                list_all_obj.append([3,c])
        
        list_obj = copy.copy(self.area_object[nb_agent][posx][posy])
        list_obj.pop(0)
        for obj in list_all_obj:
            if obj not in list_obj and obj not in objects_not_visible:
                objects_not_visible.append(obj)
        return objects_not_visible
        
    # Reset all the maps
    def reset(self):
        '''
        Reset all the arrays
        '''
        # Initialization of the world map
        # To track the agent
        self.world = []
        for nb_agent in range(self.sce_conf['nb_agents']):
            newAgent = []
            for line in range(6):
                newLine = []
                for col in range(6):
                    newLine.append(0)
                newAgent.append(newLine)
            self.world.append(newAgent)
        

        # Initialization of the area map
        # Each 0 is an area (North, South, West, East)
        # That has not been fully discovered
        self.area = []
        for nb_agent in range(self.sce_conf['nb_agents']):
            newAgent = []
            for line in range(3):
                newLine = []
                for col in range(3):
                    newLine.append(0)
                newAgent.append(newLine)
            self.area.append(newAgent)

        # Initialization of the area map with objects
        self.area_object = []
        for nb_agent in range(self.sce_conf['nb_agents']):
            newAgent = []
            for line in range(3):
                newLine = []
                for col in range(3):
                    obj = [0]
                    newLine.append(obj)
                newAgent.append(newLine)
            self.area_object.append(newAgent)


class ColorShapeMapper(ColorMapper):

    def __init__(self, sce_conf):
        """
        Mapper, set maps to track the agents and see what he sees
        :param sce_conf: (dict) parameters of the exercise
        """
        super().__init__(sce_conf)

    # Update the array of objects
    def update_area_obj(self, agent_x, agent_y, object_nb, object_color, object_shape, nb_agent):
        '''
        Update the object array of the agent depending on what it sees and where
        
        object_nb : 2 if object
                    3 if landmark

        Input: 
            agent_x: (float) Position x of the agent 
            agent_y: (float) Position y of the agent
            object_nb: (int) each object has a different number
            object_color: (int) color of the object
            object_shape: (int) shape of the object
            nb_agent: (int) number of the agent
        '''

        object = [object_nb,object_color,object_shape]
        # North
        if agent_y >= 0.33:
            if agent_x >= 0.33:
                if object not in self.area_object[nb_agent][0][2]:
                    self.area_object[nb_agent][0][2].append(object)
                    # Corner so = 2
                if self.area_object[nb_agent][0][2][0] < 2:
                    self.area_object[nb_agent][0][2][0] = 2
            elif agent_x <= -0.33:
                if object not in self.area_object[nb_agent][0][0]:
                    self.area_object[nb_agent][0][0].append(object)
                    # Corner so = 2
                if self.area_object[nb_agent][0][0][0] < 2:
                    self.area_object[nb_agent][0][0][0] = 2
            else :
                if object not in self.area_object[nb_agent][0][1]:
                    self.area_object[nb_agent][0][1].append(object)
                    # Not corner so = 1
                if self.area_object[nb_agent][0][1][0] == 0:
                    self.area_object[nb_agent][0][1][0] = 1
        # South
        elif agent_y <= -0.33:
            if agent_x >= 0.33:
                if object not in self.area_object[nb_agent][2][2]:
                    self.area_object[nb_agent][2][2].append(object)
                    # Corner so = 2
                if self.area_object[nb_agent][2][2][0] < 2:
                    self.area_object[nb_agent][2][2][0] = 2
            elif agent_x <= -0.33:
                if object not in self.area_object[nb_agent][2][0]:
                    self.area_object[nb_agent][2][0].append(object)
                # Corner so = 2
                if self.area_object[nb_agent][2][0][0] < 2:
                    self.area_object[nb_agent][2][0][0] = 2
            else :
                if object not in self.area_object[nb_agent][2][1]:
                    self.area_object[nb_agent][2][1].append(object)
                # Not corner so = 1
                if self.area_object[nb_agent][2][1][0] == 0:
                    self.area_object[nb_agent][2][1][0] = 1
        
        # Center
        else:
            if agent_x >= 0.33:
                if object not in self.area_object[nb_agent][1][2]:
                    self.area_object[nb_agent][1][2].append(object)
                # Corner so = 2
                if self.area_object[nb_agent][1][2][0] == 0:
                    self.area_object[nb_agent][1][2][0] = 1
            elif agent_x <= -0.33:
                if object not in self.area_object[nb_agent][1][0]:
                    self.area_object[nb_agent][1][0].append(object)
                # Corner so = 2
                if self.area_object[nb_agent][1][0][0] == 0:
                    self.area_object[nb_agent][1][0][0] = 1
            else :
                if object not in self.area_object[nb_agent][1][1]:
                    self.area_object[nb_agent][1][1].append(object)
                # Not corner so = 1
                if self.area_object[nb_agent][1][1][0] == 0:
                    self.area_object[nb_agent][1][1][0] = 1

    # Check an area to see if there is an object
    def check_area(self, nb_agent, direction, area_pos, all_colors, all_shapes):
        '''
        Check an selected area and return the list of object not in the area
        if obj == 2: there is an object
        if obj == 3: there is a landmark
        
        Input: 
            nb_agent: (int) number of the agent
            direction: (int) direction of the area 
            area_pos: (int) position of the area 
            all_colors: list(int) all the colors possible in the exercise
            all_shapes: list(int) all the shapes possible in the exercise

        Output:
            list_all_object: list(list(int)) return a list of object that are not in the area 
        '''
        objects = []
        list_all_obj = []
        for c in range(len(all_colors)):
            if [2,all_colors[c]] not in list_all_obj and [3,all_colors[c]] not in list_all_obj:
                list_all_obj.append([2,all_colors[c],all_shapes[c]])
                list_all_obj.append([3,all_colors[c],all_shapes[c]])
        # If North or South
        if direction == 0:
            for x in range(3) :
                objects = self.area_object[nb_agent][area_pos][x]
                for obj in objects:
                    if obj in list_all_obj:
                        list_all_obj.remove(obj)

        # If West or East
        elif direction == 1:
            for x in range(3) :
                objects = self.area_object[nb_agent][x][area_pos]
                for obj in objects:
                    if obj in list_all_obj:
                        list_all_obj.remove(obj)

        return list_all_obj

    # Return list of object not visible
    def find_missing(self, nb_agent, posx, posy, all_colors, all_shapes):
        '''
        Check an selected area and return the list of object not in the area
        if obj == 2: there is an object
        if obj == 3: there is a landmark
        
        Input: 
            nb_agent: (int) number of the agent
            posx: (int) position in the array
            posy: (int) position in the array 
            all_colors: list(int) all the colors possible in the exercise
            all_shapes: list(int) all the shapes possible in the exercise

        Output:
            list_all_object: list(list(int)) return a list of object that are not in the area 
        '''
        list_obj = []
        list_all_obj = []
        objects_not_visible = []

        # Create a list with all the possible objects
        # Each color has an object and a landmark
        for c in range(len(all_colors)):
            if [2,all_colors[c]] not in list_all_obj and [3,all_colors[c]] not in list_all_obj:
                list_all_obj.append([2,all_colors[c],all_shapes[c]])
                list_all_obj.append([3,all_colors[c],all_shapes[c]])
        
        list_obj = copy.copy(self.area_object[nb_agent][posx][posy])
        list_obj.pop(0)
        for obj in list_all_obj:
            if obj not in list_obj and obj not in objects_not_visible:
                objects_not_visible.append(obj)
        return objects_not_visible