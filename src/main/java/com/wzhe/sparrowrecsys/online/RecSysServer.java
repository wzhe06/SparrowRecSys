package com.wzhe.sparrowrecsys.online;

import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.DefaultServlet;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.util.resource.Resource;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URL;
import java.util.*;
import java.util.Map;
import java.util.stream.Collectors;


/**
 * Created by zhe.wang on 8/1/17.
 */
public class RecSysServer {

    public static void main(String[] args) throws Exception {
        new RecSysServer().run();
    }

    private static final int DEFAULT_PORT = 6010;

    public void run() throws Exception{

        int port = DEFAULT_PORT;
        try {
            port = Integer.parseInt(System.getenv("PORT"));
        } catch (NumberFormatException ex) {}


        InetSocketAddress inetAddress = new InetSocketAddress("0.0.0.0", port);
        Server server = new Server(inetAddress);

        URL webRootLocation = this.getClass().getResource("/webroot/index.html");
        if (webRootLocation == null)
        {
            throw new IllegalStateException("Unable to determine webroot URL location");
        }

        URI webRootUri = URI.create(webRootLocation.toURI().toASCIIString().replaceFirst("/index.html$","/"));
        System.err.printf("Web Root URI: %s%n",webRootUri);

        ServletContextHandler context = new ServletContextHandler();
        context.setContextPath("/");
        context.setBaseResource(Resource.newResource(webRootUri));
        context.setWelcomeFiles(new String[] { "index.html" });

        context.getMimeTypes().addMimeMapping("txt","text/plain;charset=utf-8");

        context.addServlet(DefaultServlet.class,"/");

        /*
        context.addServlet(new ServletHolder(new WeightDataServlet()), "/getdata");
        context.addServlet(new ServletHolder(new SimulationIdListServlet()), "/allsimulationids");
        context.addServlet(new ServletHolder(new MetricDataServlet()), "/metrics");
        context.addServlet(new ServletHolder(new HealthCheckServlet()), "/health_check");*/

        server.setHandler(context);

        // Add WebSocket endpoints
        //ServerContainer wsContainer = WebSocketServerContainerInitializer.configureContext(context);
        //wsContainer.addEndpoint(TimeSocket.class);

        // Add Servlet endpoints

        // Start Server
        server.start();
        server.join();
    }


    /*
    public static class SimulationIdListServlet extends HttpServlet {


        private static final long serialVersionUID = -6154475799000019575L;


        protected void doGet(HttpServletRequest request,
                             HttpServletResponse response) throws ServletException,
                IOException {

            StringBuffer result = new StringBuffer();

            try {
                response.setContentType("application/json");
                response.setStatus(HttpServletResponse.SC_OK);
                response.setCharacterEncoding("UTF-8");
                response.setHeader("Access-Control-Allow-Origin", "*");

                result.append("{\"simulation\":[");

                ArrayList<String> allSimulationId = MysqlDBService.getAllSimulationId();

                HashMap<String, String> simulationNames = MysqlDBService.getAllDockerSimulationNames();

                System.out.println("simulation id data length:\t" + allSimulationId.size());

                for (int i = 0 ; i < allSimulationId.size(); i++){

                    result.append("{\"name\":");

                    String simulationName = allSimulationId.get(i);

                    if (simulationNames.containsKey(simulationName)){
                        simulationName = simulationName + " | " + simulationNames.get(simulationName);
                    }

                    result.append("\"" + simulationName + "\"");
                    result.append(",");
                    result.append("\"value\":");
                    result.append("\"" + allSimulationId.get(i) + "\"");
                    result.append("}");

                    if (i != allSimulationId.size() - 1){
                        result.append(",");
                    }
                }

                result.append("]}");


            }catch (Exception e){
                e.printStackTrace();
                response.getWriter().println(result.toString());
            }

            //System.out.println(result.toString());
            response.getWriter().println(result.toString());

            //System.out.println("access");
        }
    }
     */
}
