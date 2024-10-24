using System;
using System.Collections.Generic;
using System.IO;
using NEA.Walker.PPO;

namespace NEA.Rendering;

// Error logger provides error messages to the user and saves it in a file.
public static class ErrorLogger
{
    public const string ErrorLogPath = "Logs/";
    public static List<string> ErrorBuffer;

    static ErrorLogger()
    {
        ErrorBuffer = new List<string>();
    }

    // Must be ran each time so that the date updates if the date changes during program runtime.
    // Get the file path of the log file per date (not per program instance), and create one if necessary.
    public static string GetLogFile()
    {
        if (!Directory.Exists(Hyperparameters.FilePath))
        {
            throw new Exception("Trying to send log to an invalid path.");
        }
        
        DateTime today = DateTime.Today;
        string todayStr = today.ToString("dd-MM-yyyy");
        string filePath = Hyperparameters.FilePath + ErrorLogPath + todayStr + ".log";
        
        if (!File.Exists(filePath))
        {
            Hyperparameters.CreateDirectories();
            using FileStream createStream = File.Create(filePath);
        }

        return filePath;
    }

    // Log an error by printing it out and store the log in the log file.
    public static void LogError(string error)
    {
        LogMessage($"ERROR: {error}");
    }

    // Log a warning by printing it out and store the log in the log file.
    public static void LogWarning(string warning)
    {
        LogMessage($"WARNING: {warning}");
    }

    // Log a message by printing it out and store the log in the log file.
    private static void LogMessage(string message)
    {
        if (message == "")
        {
            LogError("Log called with empty message.");
            return;
        }

        ErrorBuffer.Add(message);
        message += "\n";
        Console.WriteLine(message);

        string filePath;
        try
        {
            filePath = GetLogFile();
        }
        catch (Exception e)
        {
            Console.WriteLine("ERROR: File path invalid, cannot log error.");
            return;
        }
        
        File.AppendAllText(filePath, $"{DateTime.Now.ToString()} {message}");
    }
}