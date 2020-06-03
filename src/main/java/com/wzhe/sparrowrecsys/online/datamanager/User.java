package com.wzhe.sparrowrecsys.online.datamanager;

import java.util.ArrayList;
import java.util.List;

public class User {
    int userId;
    List<Rating> ratings;

    public User(){
        this.ratings = new ArrayList<>();
    }

    public int getUserId() {
        return userId;
    }

    public void setUserId(int userId) {
        this.userId = userId;
    }

    public List<Rating> getRatings() {
        return ratings;
    }

    public void setRatings(List<Rating> ratings) {
        this.ratings = ratings;
    }

    public void addRating(Rating rating) {
        this.ratings.add(rating);
    }
}
