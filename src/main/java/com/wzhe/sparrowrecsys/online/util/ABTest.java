package com.wzhe.sparrowrecsys.online.util;

public class ABTest {
    final static int trafficSplitNumber = 5;

    final static String bucketAModel = "emb";
    final static String bucketBModel = "nerualcf";

    final static String defaultModel = "emb";

    public static String getConfigByUserId(String userId){
        if (null == userId || userId.isEmpty()){
            return defaultModel;
        }

        if(userId.hashCode() % trafficSplitNumber == 0){
            System.out.println(userId + " is in bucketA.");
            return bucketAModel;
        }else if(userId.hashCode() % trafficSplitNumber == 1){
            System.out.println(userId + " is in bucketB.");
            return bucketBModel;
        }else{
            System.out.println(userId + " isn't in AB test.");
            return defaultModel;
        }
    }
}
