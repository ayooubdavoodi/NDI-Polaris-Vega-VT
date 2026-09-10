//!  @file main.cpp The Combined API (CAPI) sample application.
//----------------------------------------------------------------------------
//
//  Copyright (C) 2017, Northern Digital Inc. All rights reserved.
//
//  All Northern Digital Inc. ("NDI") Media and/or Sample Code and/or Sample Code
//  Documentation (collectively referred to as "Sample Code") is licensed and provided "as
//  is" without warranty of any kind. The licensee, by use of the Sample Code, warrants to
//  NDI that the Sample Code is fit for the use and purpose for which the licensee intends to
//  use the Sample Code. NDI makes no warranties, express or implied, that the functions
//  contained in the Sample Code will meet the licensee's requirements or that the operation
//  of the programs contained therein will be error free. This warranty as expressed herein is
//  exclusive and NDI expressly disclaims any and all express and/or implied, in fact or in
//  law, warranties, representations, and conditions of every kind pertaining in any way to
//  the Sample Code licensed and provided by NDI hereunder, including without limitation,
//  each warranty and/or condition of quality, merchantability, description, operation,
//  adequacy, suitability, fitness for particular purpose, title, interference with use or
//  enjoyment, and/or non infringement, whether express or implied by statute, common law,
//  usage of trade, course of dealing, custom, or otherwise. No NDI dealer, distributor, agent
//  or employee is authorized to make any modification or addition to this warranty.
//  In no event shall NDI nor any of its employees be liable for any direct, indirect,
//  incidental, special, exemplary, or consequential damages, sundry damages or any
//  damages whatsoever, including, but not limited to, procurement of substitute goods or
//  services, loss of use, data or profits, or business interruption, however caused. In no
//  event shall NDI's liability to the licensee exceed the amount paid by the licensee for the
//  Sample Code or any NDI products that accompany the Sample Code. The said limitations
//  and exclusions of liability shall apply whether or not any such damages are construed as
//  arising from a breach of a representation, warranty, guarantee, covenant, obligation,
//  condition or fundamental term or on any theory of liability, whether in contract, strict
//  liability, or tort (including negligence or otherwise) arising in any way out of the use of
//  the Sample Code even if advised of the possibility of such damage. In no event shall
//  NDI be liable for any claims, losses, damages, judgments, costs, awards, expenses or
//  liabilities of any kind whatsoever arising directly or indirectly from any injury to person
//  or property, arising from the Sample Code or any use thereof
//
//----------------------------------------------------------------------------

#ifdef _WIN32
#define ACCESS _access
#include <conio.h>   // for _kbhit()
#include <io.h>      // for _access_s()
#include <windows.h> // for Sleep(ms)
#else
#define ACCESS access
#include <unistd.h>     // for POSIX sleep(sec), and access()
#include <sys/ioctl.h>
#endif

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>

#include "CombinedApi.h"
#include "PortHandleInfo.h"
#include "ToolData.h"

static CombinedApi capi = CombinedApi();
static bool apiSupportsBX2 = false;
static bool apiSupportsStreaming = false;
static bool useEncryption = false;
static std::string cipher = "";
static bool useUDP = false;

/*
 * @brief Equivalent of std::filesystem::exists() that works with -std=c++11
 */
bool fileExists(const std::string &Filename)
{
    return ACCESS(Filename.c_str(), 0) == 0;
}

/**
 * @brief There's no standard cross platform sleep() keystroke detection
 */
int detectKeystroke()
{
#ifdef _WIN32
    return _kbhit();
#else
    int bytesWaiting;
    ioctl(0, FIONREAD, &bytesWaiting); //stdin=0
    return bytesWaiting;
#endif
}

/**
 * @brief There's no standard cross platform sleep() method prior to C++11
 */
void sleepSeconds(unsigned numSeconds)
{
#ifdef _WIN32
    Sleep((DWORD)1000 * numSeconds); // Sleep(ms)
#else
    sleep(numSeconds); // sleep(sec)
#endif
}

/**
 * @brief Prints a debug message if a method call failed.
 * @details To use, pass the method name and the error code returned by the method.
 *          Eg: onErrorPrintDebugMessage("capi.initialize()", capi.initialize());
 *          If the call succeeds, this method does nothing.
 *          If the call fails, this method prints an error message to stdout.
 */
void onErrorPrintDebugMessage(std::string methodName, int errorCode)
{
    if (errorCode < 0)
    {
        std::cout << methodName << " failed: " << capi.errorToString(errorCode) << std::endl;
    }
}

/**
* @brief Returns the string: "[tool.id] s/n:[tool.serialNumber]" used in CSV output
*/
std::string getToolInfo(std::string toolHandle)
{
    // Get the port handle info from PHINF
    PortHandleInfo info = capi.portHandleInfo(toolHandle);

    // Return the ID and SerialNumber the desired string format
    std::string outputString = info.getToolId();
    outputString.append(" s/n:").append(info.getSerialNumber());
    return outputString;
}

/**
* @brief Returns a string representation of the data in CSV format.
* @details The CSV format is: "Frame#,ToolHandle,Face,TransformStatus,q0,qx,qy,qz,tx,ty,tz,error,#markers,[Marker1:status,x,y,z],[Marker2..."
*/
std::string toolDataToCSV(const ToolData& toolData)
{
    std::stringstream stream;
    stream << std::setprecision(toolData.PRECISION) << std::setfill('0');
    stream << "" << static_cast<unsigned>(toolData.frameNumber) << ","
           << "Port:" << static_cast<unsigned>(toolData.transform.toolHandle) << ",";
    stream << static_cast<unsigned>(toolData.transform.getFaceNumber()) << ",";

    if (toolData.transform.isMissing())
    {
        stream << "Missing,,,,,,,,";
    }
    else
    {
        stream << TransformStatus::toString(toolData.transform.getErrorCode()) << ","
               << toolData.transform.q0 << "," << toolData.transform.qx << "," << toolData.transform.qy << "," << toolData.transform.qz << ","
               << toolData.transform.tx << "," << toolData.transform.ty << "," << toolData.transform.tz << "," << toolData.transform.error;
    }

    // Each marker is printed as: status,tx,ty,tz
    stream << "," << toolData.markers.size();
    for ( int i = 0; i < toolData.markers.size(); i++)
    {
        stream << "," << MarkerStatus::toString(toolData.markers[i].status);
        if (toolData.markers[i].status == MarkerStatus::Missing)
        {
            stream << ",,,";
        }
        else
        {
            stream << "," << toolData.markers[i].x << "," << toolData.markers[i].y << "," << toolData.markers[i].z;
        }
    }
    return stream.str();
}

/**
 * @brief Write tracking data to a CSV file in the format: "#Tools,ToolInfo,Frame#,[Tool1],Frame#,[Tool2]..."
 * @details It's worth noting that the number lines in the file does not necessarily match the number of frames collected.
 *          NDI measurement systems support different types of tools: passive, active, and active-wireless.
 *          Because different tool types are detected in different physical ways, each tool type has a separate frame for
 *          collecting data for all tools of that type. Each line of the file has the same number of tools, but each tool
 *          may have a different frame number that corresponds to its tool type.
 * @param fileName The file to write to.
 * @param numberOfLines The number of lines to write.
 * @param enabledTools ToolData storing the serial number for each enabled tool.
 */
void writeCSV(std::string fileName, int numberOfLines, std::vector<ToolData>& enabledTools)
{
    // Print header information to the first line of the output file
    std::cout << std::endl << "Writing CSV file..." << std::endl;
    std::ofstream csvFile(fileName.c_str());
    csvFile << "#Tools";

    // Loop to gather tracking data and write to the file
    int linesWritten = 0;
    int previousFrameNumber = 0; // use this variable to avoid printing duplicate data with BX
    while (linesWritten < numberOfLines)
    {
        // Get new tool data using BX2
        std::vector<ToolData> newToolData = apiSupportsBX2 ? capi.getTrackingDataBX2("--6d=tools --3d=tools --sensor=none --1d=buttons") :
                                                             capi.getTrackingDataBX(TrackingReplyOption::TransformData | TrackingReplyOption::AllTransforms);

        // Update enabledTools array with new data
        for (int t = 0; t < enabledTools.size(); t++)
        {
            for (int j = 0; j < newToolData.size(); j++)
            {
                if (enabledTools[t].transform.toolHandle == newToolData[j].transform.toolHandle)
                {
                    // Copy the new tool data
                    newToolData[j].toolInfo = enabledTools[t].toolInfo; // keep the serial number
                    enabledTools[t] = newToolData[j]; // use the new data
                }
            }
        }

        // If we're using BX2 there's extra work to do because BX2 and BX use opposite philosophies.
        // BX will always return data for all enabled tools, but some of the data may be old: #linesWritten == # BX commands
        // BX2 never returns old data, but cannot guarantee new data for all enabled tools with each call: #linesWritten <= # BX2 commands
        // We want a CSV file with data for all enabled tools in each line, but this requires multiple BX2 calls.
        if (apiSupportsBX2)
        {
            // Count the number of tools that have new data
            int newDataCount = 0;
            for (int t = 0; t < enabledTools.size(); t++)
            {
                if (enabledTools[t].dataIsNew)
                {
                    newDataCount++;
                }
            }

            // Send another BX2 if some tools still have old data
            if (newDataCount < enabledTools.size())
            {
                continue;
            }
        }
        else
        {
            if (previousFrameNumber == enabledTools[0].frameNumber)
            {
                // If the frame number didn't change, don't print duplicate data to the CSV, send another BX
                continue;
            }
            else
            {
                // This frame number is different, so we'll print a line to the CSV, but remember it for next time
                previousFrameNumber = enabledTools[0].frameNumber;
            }
        }

        // If this is the first line of the CSV, print labels for tool and marker data
        if (linesWritten == 0)
        {
            for (int t = 0; t < enabledTools.size(); t++)
            {
                csvFile << ",ToolInfo,Frame#,PortHandle,Face#,TransformStatus,Q0,Qx,Qy,Qz,Tx,Ty,Tz,Error,#Markers";
                for (int m = 0; m < enabledTools[t].markers.size(); m++)
                {
                    csvFile << ",Marker" << m << ".Status,Tx,Ty,Tz";
                }
            }
            csvFile << std::endl;
        }


        // Print a line of the CSV file if all enabled tools have new data
        csvFile << std::dec << enabledTools.size();
        for (int t = 0; t < enabledTools.size(); t++)
        {
            csvFile << "," << enabledTools[t].toolInfo << "," << toolDataToCSV(enabledTools[t]);
            enabledTools[t].dataIsNew = false; // once printed, the data becomes "old"
        }
        csvFile << std::endl;
        linesWritten++;
    }
}

/**
 * @brief Prints a ToolData object to stdout
 * @param toolData The data to print
 */
void printToolData(const ToolData& toolData)
{
    if (toolData.systemAlerts.size() > 0)
    {
        std::cout << "[" << toolData.systemAlerts.size() << " alerts] ";
        for (int a = 0; a < toolData.systemAlerts.size(); a++)
        {
            std::cout << toolData.systemAlerts[a].toString() << std::endl;
        }
    }

    if (toolData.buttons.size() > 0)
    {
        std::cout << "[buttons: ";
        for (int b = 0; b < toolData.buttons.size(); b++)
        {
            std::cout << ButtonState::toString(toolData.buttons[b]) << " ";
        }
        std::cout << "] ";
    }
    std::cout << toolDataToCSV(toolData) << std::endl;
}

/**
 * @brief Put the system into tracking mode, and poll a few frames of data.
 */
void printTrackingData()
{
    // Start tracking, output a few frames of data, and stop tracking

    for (int i = 0; i < 10; i++)
    {
        // Demonstrate TX command: ASCII command sent, ASCII reply received
        std::cout << capi.getTrackingDataTX() << std::endl;

        // Demonstrate BX or BX2 command
        std::vector<ToolData> toolData =  apiSupportsBX2 ? capi.getTrackingDataBX2() : capi.getTrackingDataBX();

        // Print to stdout in similar format to CSV
        std::cout << "[alerts] [buttons] Frame#,ToolHandle,Face#,TransformStatus,Q0,Qx,Qy,Qz,Tx,Ty,Tz,Error,#Markers,State,Tx,Ty,Tz" << std::endl;
        for (int i = 0; i < toolData.size(); i++)
        {
            printToolData(toolData[i]);
        }
    }
}

/* The streamed reply, may be used to quit streaming if there is an ERROR */
std::string streamedReply = "";
std::string input_string  = "<<0101+00062-07162+06968-00367+007031-003674-102789+0334600000231479199810000";

/*
 * @brief Streams a command using insecure or secure TCP
 */
void streamTcp()
{
    std::string streamId = useEncryption ? "tlsStream1" : "tcpStream1";

    std::cout << std::endl << "**********Here 1*********" << std::endl;
    capi.startStreaming("TX 0801", streamId, useEncryption ? Protocol::SecureTCP : Protocol::TCP, cipher);

    // Wait until a keystroke, or if there's an "ERROR" in the stream
    while (detectKeystroke() == 0 && streamedReply.rfind( "ERROR", 0 ) != 0 )
    {
        streamedReply = capi.readStream(streamId);
        std::cout << std::endl << "**********Here 3*********" << std::endl;
        input_string =streamedReply;

        // Remove leading "<<"
        if (input_string.rfind("<<", 0) == 0) {
            input_string = input_string.substr(2);
        }

        // First 4 chars = handle
        std::string handle = input_string.substr(0,  4);
        std::string q0_str = input_string.substr(4,  6);
        std::string qx_str = input_string.substr(10, 6);
        std::string qy_str = input_string.substr(16, 6);
        std::string qz_str = input_string.substr(22, 6);
        std::string tx_str = input_string.substr(28, 7);
        std::string ty_str = input_string.substr(35, 7);
        std::string tz_str = input_string.substr(42, 7);
        std::string er_str = input_string.substr(49, 6);

        // std::cout << "Handle: " << handle << std::endl;
        // std::cout << "q0_str: " << q0_str << std::endl;
        // std::cout << "qx_str: " << qx_str << std::endl;
        // std::cout << "qy_str: " << qy_str << std::endl;
        // std::cout << "qz_str: " << qz_str << std::endl;
        // std::cout << "tx_str: " << tx_str << std::endl;
        // std::cout << "ty_str: " << ty_str << std::endl;
        // std::cout << "tz_str: " << tz_str << std::endl;
        // std::cout << "er_str: " << er_str << std::endl;

        double q0 = std::stoi(q0_str) / 10000.0;
        double qx = std::stoi(qx_str) / 10000.0;
        double qy = std::stoi(qy_str) / 10000.0;
        double qz = std::stoi(qz_str) / 10000.0;
        double tx = std::stoi(tx_str) / 100.0; // if 0.01mm units
        double ty = std::stoi(ty_str) / 100.0;
        double tz = std::stoi(tz_str) / 100.0;
        double er = std::stoi(er_str) / 10000.0;

        // std::cout << "Quaternion: (" << q0 << ", " << qx << ", " << qy << ", " << qz << ")" << std::endl;
        // std::cout << "Translation: (" << tx << ", " << ty << ", " << tz << ")" << std::endl;
        // std::cout << "Error: " << er << std::endl;

    }
    capi.stopStreaming(streamId, useEncryption ? Protocol::SecureTCP : Protocol::TCP );
    std::cout << std::endl << "**********Here 2*********" << std::endl;

}

/*
 * @brief Streams a command using insecure or secure UDP
 */
void streamUdp()
{
    std::string streamId = useEncryption ? "dtlsStream1" : "udpStream1";
    capi.startStreaming("TX 0801", streamId, useEncryption ? Protocol::SecureUDP : Protocol::UDP, cipher);

    // Wait until a keystroke, or if there's an "ERROR" in the stream
    while ( detectKeystroke() == 0 && streamedReply.rfind( "ERROR", 0 ) != 0 )
    {
        streamedReply = capi.readStream(streamId);
    }
    capi.stopStreaming(streamId, useEncryption ? Protocol::SecureUDP : Protocol::UDP );
}

/**
 * @brief Initialize and enable loaded tools. This is the same regardless of tool type.
 */
void initializeAndEnableTools(std::vector<ToolData>& enabledTools)
{
    std::cout << std::endl << "Initializing and enabling tools..." << std::endl;

    // Initialize and enable tools
    std::vector<PortHandleInfo> portHandles = capi.portHandleSearchRequest(PortHandleSearchRequestOption::NotInit);
    for (int i = 0; i < portHandles.size(); i++)
    {
        onErrorPrintDebugMessage("capi.portHandleInitialize()", capi.portHandleInitialize(portHandles[i].getPortHandle()));
        onErrorPrintDebugMessage("capi.portHandleEnable()", capi.portHandleEnable(portHandles[i].getPortHandle()));
    }

    // Print all enabled tools
    portHandles = capi.portHandleSearchRequest(PortHandleSearchRequestOption::Enabled);
    for (int i = 0; i < portHandles.size(); i++)
    {
        std::cout << portHandles[i].toString() << std::endl;
    }

    // Lookup and store the serial number for each enabled tool
    for (int i = 0; i < portHandles.size(); i++)
    {
        enabledTools.push_back(ToolData());
        enabledTools.back().transform.toolHandle = (uint16_t)capi.stringToInt(portHandles[i].getPortHandle());
        enabledTools.back().toolInfo = getToolInfo(portHandles[i].getPortHandle());
    }
}

/**
 * @brief Loads a tool from a tool definition file (.rom)
 */
void loadTool(const char* toolDefinitionFilePath)
{
    // Request a port handle to load a passive tool into
    int portHandle = capi.portHandleRequest();
    onErrorPrintDebugMessage("capi.portHandleRequest()", portHandle);

    // Load the .rom file using the previously obtained port handle
    capi.loadSromToPort(toolDefinitionFilePath, portHandle);
}

/**
 * @brief Demonstrate detecting active tools.
 * @details Active tools are connected through a System Control Unit (SCU) with physical wires.
 */
void configureActiveTools(std::string scuHostname)
{
    // Setup the SCU connection for demonstrating active tools
    std::cout << std::endl << "Configuring Active Tools - Setup SCU Connection" << std::endl;
    onErrorPrintDebugMessage("capi.setUserParameter()", capi.setUserParameter("Param.Connect.SCU Hostname", scuHostname));
    std::cout << capi.getUserParameter("Param.Connect.SCU Hostname") << std::endl;

    // Wait a few seconds for the SCU to detect any wired tools plugged in
    std::cout << std::endl << "Demo Active Tools - Detecting Tools..." << std::endl;
    sleepSeconds(2);

    // Print all port handles
    std::vector<PortHandleInfo> portHandles = capi.portHandleSearchRequest(PortHandleSearchRequestOption::NotInit);
    for (int i = 0; i < portHandles.size(); i++)
    {
        std::cout << portHandles[i].toString() << std::endl;
    }
}

/**
 * @brief Demonstrate loading an active wireless tool.
 * @details Active wireless tools are battery powered and emit IR in response to a chirp from the illuminators.
 */
void configureActiveWirelessTools()
{
    // Load an active wireless tool definitions from a .rom files
    std::cout << std::endl << "Configuring an Active Wireless Tool - Loading .rom File..." << std::endl;
    loadTool("sroms/active-wireless.rom");
}

/**
 * @brief Demonstrate loading dummy tools of each tool type.
 * @details Dummy tools are used to report 3Ds in the absence of real tools.
 *          Dummy tools should not be loaded with regular tools of the same type.
 *          TSTART will fail if real and dummy tools are enabled simultaneously.
 */
void configureDummyTools()
{
    std::cout << std::endl << "Loading passive, active-wireless, and active dummy tools..." << std::endl;
    onErrorPrintDebugMessage("capi.loadPassiveDummyTool()", capi.loadPassiveDummyTool());
    onErrorPrintDebugMessage("capi.loadActiveWirelessDummyTool()", capi.loadActiveWirelessDummyTool());
    onErrorPrintDebugMessage("capi.loadActiveDummyTool()", capi.loadActiveDummyTool());
}

/**
 * @brief Demonstrate getting/setting user parameters.
 */
void configureUserParameters()
{
    std::cout << capi.getUserParameter("Param.User.String0") << std::endl;
    onErrorPrintDebugMessage("capi.setUserParameter(Param.User.String0, customString)", capi.setUserParameter("Param.User.String0", "customString"));
    std::cout << capi.getUserParameter("Param.User.String0") << std::endl;
    onErrorPrintDebugMessage("capi.setUserParameter(Param.User.String0, emptyString)", capi.setUserParameter("Param.User.String0", ""));
}

/**
 * @brief Sets the user parameter "Param.Simulated Alerts" to test communication of system alerts.
 * @details This method does nothing if simulatedAlerts is set to 0x00000000.
 */
void simulateAlerts(uint32_t simulatedAlerts = 0x00000000)
{
    // Simulate alerts if any were requested
    if (simulatedAlerts > 0x0000)
    {
        std::cout << std::endl << "Simulating system alerts..." << std::endl;
        std::stringstream stream;
        stream << simulatedAlerts;
        onErrorPrintDebugMessage("capi.setUserParameter(Param.Simulated Alerts, alerts)", capi.setUserParameter("Param.Simulated Alerts", stream.str()));
        std::cout << capi.getUserParameter("Param.Simulated Alerts") << std::endl;
    }
}

/**
 * @brief Determines whether an NDI device supports the BX2 command by looking at the API revision
 */
void determineApiSupportForBX2()
{
    // Lookup the API revision
    std::string response = capi.getApiRevision();

    // Refer to the API guide for how to interpret the APIREV response
    char deviceFamily = response[0];
    int majorVersion = capi.stringToInt(response.substr(2,3));

    // As of early 2017, the only NDI device supporting BX2 is the Vega
    // Vega is a Polaris device with API major version 003
    if ( deviceFamily == 'G' && majorVersion >= 3)
    {
        apiSupportsBX2 = true;
        apiSupportsStreaming = true;
    }
}

void printHelp()
{
    std::cout << "CAPIsample Ver " << capi.getVersion() << std::endl
        << "usage: ./capisample <hostname> --tools=[file1.rom],[file2.rom] [args]" << std::endl
        << "or   : ./capisample <input gbf file> <output txt file>" << std::endl
        << "where:" << std::endl
        << "    <hostname>      The measurement device's hostname, IP address, or serial port." << std::endl
        << "    --tools=[file1.rom],[file2.rom]...  A comma delimited list of tools to load." << std::endl
        << "    [args]          (optional) Any other arguments such as tools to load, or SCU to connect to." << std::endl
        << "    <input gbf file> a file that contains GBF data recorded by ToolBox." << std::endl
        << "    <output txt file> destination output file to contain the text representation of the input GBF file data." << std::endl
        << "example hostnames:" << std::endl
        << "    Connecting to device by IP address: 169.254.8.50" << std::endl
        << "    Connecting to device by hostname: P9-B0103.local" << std::endl
        << "    Connecting to serial port varies by operating system:" << std::endl
        << "        COM10 (Windows), /dev/ttyUSB0 (Linux), /dev/cu.usbserial-001014FA (Mac)" << std::endl
        << "Optional arguments:" << std::endl
#ifdef OPENSSL
        << "    --cipher=[ciphersuite] Limit OpenSSL to use a single TLS/DTLS ciphersuite. Only valid when --encrypted argument is also passed. For example --cipher=ECDHE-RSA-CAMELLIA256-SHA384." << std::endl
        << "    --encrypted Encrypt communcation with the device. Can specify ciphersuite with --cipher but not required." << std::endl
#endif
        << "    --help Displays the help information." << std::endl
        << "    --scu=[scu_hostname] A System Control Unit (SCU) hostname, used to connect active tools." << std::endl
        << "    --udp Stream data over UDP instead of TCP (default)" << std::endl;
}

/**
 * @brief   The entry point for the CAPIsample application.
 * @details The invocation of CAPIsample is expected to pass a few arguments: ./CAPIsample [hostname] [args]
 *          arg(0) - (default)  The path to this application (ignore this)
 *          arg(1) - (required) The measurement device's hostname, IP address, or serial port.
 *          Eg: Connecting to device by IP address: "169.254.8.50"
 *          Eg: Connecting to device by zeroconf hostname: "P9-B0103.local"
 *          Eg: Connecting to serial port varies by OS: "COM10" (Win), "/dev/ttyUSB0" (Linux), "/dev/cu.usbserial-001014FA" (Mac)
 *          arg(2+) - (optional) Specify options such as: --encrypted, --scu=[scu_hostname], --tools=[file1.rom],[file2.rom]
 */
int main(int argc, char* argv[])
{
    // Print help information if the user has not specified any arguments
    if (argc < 2)
    {
        printHelp();
        return -1;
    }

    std::string inputName = argv[1];

    // if doesn't start with COM or dev
    if ( inputName.substr( 0, 3 ).compare( "COM" ) != 0 && inputName.substr( 0, 4 ).compare( "/dev" ) != 0 )
    {
        // Instead of a hostname, argv[1] may be a path to a GBF file to convert
        std::ifstream input( argv[1], std::ios::in | std::ios::binary );
        if ( input )
        {
            input.close();
            if ( argc == 3 )
            {
                if ( capi.convertGbfFileToText( argv[1], argv[2] ) == 0 )
                {
                    std::cout << "Saved file to " << argv[2] << " successfully." << std::endl;
                    // quit
                    return 0;
                }
                else
                {
                    // Failure
                    return -1;
                }
            }
            else
            {
                std::cerr << "GBF-to-text conversion requires an input and an output file to be specified." << std::endl;
                return -1;
            }
        }
    }

    // Assign the hostname
    std::string hostname = std::string( argv[1] );

    // Look for optional arguments
    std::string scu_hostname = "";
    std::vector<std::string> toolDefinitions = std::vector<std::string>();
    for (int i = 2; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (arg.substr(0, 6).compare("--scu=") == 0)
        {
            scu_hostname = arg;
        }
        else if (arg.substr(0, 11).compare("--encrypted") == 0)
        {
            useEncryption = true;
        }
        else if (arg.substr(0, 9).compare("--cipher=") == 0)
        {
            cipher = arg.substr(9);
        }
        else if (arg.substr(0, 5).compare("--udp") == 0)
        {
            useUDP = true;
        }
        else if (arg.substr(0, 8).compare("--tools=") == 0)
        {
            std::istringstream sstream(arg.substr(8));
            std::string tool;
            while (getline(sstream, tool, ','))
            {
                // If the file is accessible, add it to the list of tools to load
                if (fileExists(tool))
                {
                    toolDefinitions.push_back(tool);
                }
                else
                {
                    // If a tool file cannot be accessed then exit
                    std::cerr << "Cannot access file: " << tool << std::endl;
                    return -1;
                }
            }
        }
        else if (arg.substr(0, 6).compare("--help") == 0)
        {
            printHelp();
            return -1;
        }
        else
        {
            std::cerr << "Unrecognized option: " << arg << std::endl;
            return -1;
        }
    }

    if((useEncryption == false) && (cipher != ""))
    {
        std::cerr << "Invalid arguments: A ciphersuite cannot be specified unless the --encrypted option is also passed" << std::endl;
        return -1;
    }

    // Open the command channel to the device
    if (capi.connect(hostname, useEncryption ? Protocol::SecureTCP : Protocol::TCP, cipher) != 0)
    {
        // Print the error and exit if we can't connect to a device
        std::cout << "Connection Failed!" << std::endl;
        std::cout << "Press Enter to continue...";
        std::cin.ignore();
        return -1;
    }
    std::cout << "Connected!" << std::endl;

    // Wait a second - needed to support connecting to LEMO Vega
    sleepSeconds(1);

    // Print the firmware version for debugging purposes
    std::cout << capi.getUserParameter("Features.Firmware.Version") << std::endl;

    // Determine if the connected device supports the BX2 command
    determineApiSupportForBX2();

    // Initialize the system. This clears all previously loaded tools, unsaved settings etc...
    onErrorPrintDebugMessage("capi.initialize()", capi.initialize());

    // Demonstrate error handling by asking for tracking data in the wrong mode
    std::cout << capi.getTrackingDataTX() << std::endl;

    // Demonstrate getting/setting user parameters
    configureUserParameters();

    // Load any passive tool definitions from a .rom files
    if (toolDefinitions.size() > 0)
    {
        std::cout << "Loading Tool Definitions (.rom files) ..." << std::endl;
        for (int f = 0; f < toolDefinitions.size(); f++)
        {
            std::cout << "Loading: " << toolDefinitions[f] << std::endl;
            loadTool(toolDefinitions[f].c_str());
        }
    }

    // Wired tools are connected through a System Control Unit (SCU)
    if (scu_hostname.length() > 0)
    {
        configureActiveTools(scu_hostname);
    }

    // Dummy tools are used to report 3Ds in the absence of real tools.
    // TSTART will fail if real and dummy tools of the same type are enabled simultaneously.
    // To experiment with dummy tools, comment out the previous tool configurations first.
    // configureDummyTools();

    // Once loaded or detected, tools are initialized and enabled the same way
    // PHSR can be time consuming, so store the tool metadata immediately (eg. port handle, serial number)
    std::vector<ToolData> enabledTools = std::vector<ToolData>();
    initializeAndEnableTools(enabledTools);

    // Print an error if no tools were specified
    if (enabledTools.size() == 0)
    {
        std::cout << "No tools detected. To load passive tools, specify: --tools=[tool1.rom],[tool2.rom]" << std::endl;
    }

    // Spoofing system faults is handy to see how system alerts are handled, but faults can
    // prevent tracking data from being returned. This sample application has valued code simplicity
    // above robustness, so it does not gracefully handle every error case. You can experiment with
    // system faults, but attempting to do other tasks (eg. writing a .csv of tracking data) may fail
    // depending on what faults have been simulated.
    // simulateAlerts(0x7FFFFFFF);

    std::cout << std::endl << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

    // Once the system is put into tracking mode, data is returned for whatever tools are enabled
    std::cout << std::endl << "Entering tracking mode..." << std::endl;
    onErrorPrintDebugMessage("capi.startTracking()", capi.startTracking());

    if (!apiSupportsStreaming)
    {
        
        // Demonstrate polling a few frames of data
        printTrackingData();
        std::cout << std::endl << "START streaming Kia" << std::endl;

        // Write a CSV file
        writeCSV("example.csv", 50, enabledTools);
    }
    else
    {
        // Stream tracking data to the terminal on another thread

        std::cout << std::endl << "------------------------------------------" << std::endl;
        std::thread streamingThread(useUDP ? streamUdp : streamTcp);

        // Command/response is possible while data is streaming.
        // Keep the command channel alive by issuing commands.
        // Param.Connect.Idle Timeout=300s by default will terminate the connection if there is no activity.
        // If there's an ERROR in the streamed replies, quit out of the polling loop
        while (detectKeystroke() == 0 && streamedReply.rfind( "ERROR", 0 ) != 0 )
        {
            // Periodically read the gravity vector
            sleepSeconds(1);
            capi.getUserParameter("Info.Status.Gravity Vector");
        }

        // Wait for the streaming thread to stop and cleanup
        streamingThread.join();
        std::cout << std::endl << "------------------------------------------" << std::endl;
    }
    std::cout << std::endl << "------------------------------------------" << std::endl;

    // Stop tracking (back to configuration mode)
    std::cout << std::endl << "Leaving tracking mode and returning to configuration mode..." << std::endl;
    onErrorPrintDebugMessage("capi.stopTracking()", capi.stopTracking());

    std::cout << "CAPI demonstration complete. Press any key to exit." << std::endl;
    return 0;
}