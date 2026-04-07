# VxWorks Basics

## Introduction

VxWorks is a real-time operating system (RTOS) developed by Wind River Systems, widely used in embedded systems requiring deterministic behavior and high reliability. This module introduces you to the fundamentals of VxWorks and the Wind River Workbench development environment.

VxWorks is designed for mission-critical applications in aerospace, defense, automotive, industrial automation, and telecommunications industries. Its key strengths include:

- **Deterministic real-time performance**: Guaranteed response times for critical operations
- **Scalability**: From small microcontrollers to multi-core processors
- **Safety and security**: Certified for safety-critical applications
- **POSIX compliance**: Standard API support for portability

## Prerequisites

Before starting this module, ensure you have:

- **C Programming Experience**: At least one year of C programming experience
- **Operating Systems Knowledge**: Basic understanding of operating system concepts (processes, threads, memory management)
- **Debugging Skills**: Familiarity with basic debugging techniques
- **Development Environment**: Access to Wind River Workbench 4 and VxWorks 22.06 or later

## Core Concepts

### 1. VxWorks Architecture Overview

VxWorks follows a monolithic kernel architecture where applications run in kernel space by default, providing maximum performance and minimal overhead.

```c
/* Basic VxWorks application structure */
#include <vxWorks.h>
#include <taskLib.h>
#include <stdio.h>

/* Entry point for VxWorks application */
int myApp(void)
{
    printf("Hello VxWorks!\n");
    return OK;
}
```

**Key Components:**
- **Kernel**: Core RTOS functionality
- **Wind Kernel**: Microkernel providing basic services
- **I/O System**: Device drivers and file systems
- **Network Stack**: TCP/IP and other protocols

### 2. Wind River Workbench Environment

Workbench is the integrated development environment (IDE) for VxWorks development.

**Main Features:**
- Project management and build system
- Source code editor with syntax highlighting
- Integrated debugger
- Target connection management
- System analysis tools

**Project Types:**
```bash
# VxWorks Image Project (VIP)
# Contains kernel configuration and boot image

# Downloadable Kernel Module (DKM)
# Kernel-space applications

# Real-Time Process (RTP)
# User-space applications with memory protection
```

### 3. VxWorks Kernel Shell

The kernel shell provides command-line interface for system interaction and debugging.

```bash
# Basic shell commands
-> help                    # Display available commands
-> i                      # Show task information
-> memShow                # Display memory statistics
-> devs                   # List devices
-> ld < myApp.out         # Load object module
```

**Common Shell Operations:**
```bash
# Task management
-> sp myTask              # Spawn a task
-> td "myTask"            # Delete a task
-> ts "myTask"            # Suspend a task
-> tr "myTask"            # Resume a task

# Memory operations
-> d 0x1000000            # Display memory contents
-> m 0x1000000            # Modify memory
-> checkStack "myTask"    # Check task stack usage
```

### 4. Real-Time Tasks

Tasks are the fundamental execution units in VxWorks.

```c
#include <vxWorks.h>
#include <taskLib.h>
#include <stdio.h>

/* Task function */
void myTask(int arg1, int arg2, int arg3, int arg4, int arg5)
{
    printf("Task started with args: %d, %d\n", arg1, arg2);
    
    /* Task main loop */
    FOREVER
    {
        printf("Task running...\n");
        taskDelay(sysClkRateGet()); /* Delay 1 second */
    }
}

/* Spawn a task */
int createMyTask(void)
{
    TASK_ID taskId;
    
    taskId = taskSpawn("tMyTask",           /* Task name */
                       100,                 /* Priority (0-255) */
                       VX_FP_TASK,         /* Options */
                       8192,               /* Stack size */
                       (FUNCPTR)myTask,    /* Entry point */
                       1, 2, 0, 0, 0);     /* Arguments */
    
    if (taskId == TASK_ID_ERROR)
    {
        printf("Failed to create task\n");
        return ERROR;
    }
    
    return OK;
}
```

**Task States:**
- **READY**: Ready to run
- **PEND**: Waiting for resource
- **DELAY**: Sleeping for specified time
- **SUSPEND**: Explicitly suspended

### 5. Target Configuration and Connection

VxWorks applications run on target hardware or simulators.

**Target Connection Setup:**
```bash
# Configure target connection in Workbench
# 1. Create new target connection
# 2. Specify target type (hardware/simulator)
# 3. Configure communication method (Ethernet, serial, etc.)
# 4. Set target IP address and connection parameters
```

**Boot Configuration:**
```c
/* bootline example for network boot */
"mottsec(0,0)host:/path/to/vxWorks h=192.168.1.100 e=192.168.1.10 u=user"

/* Where: */
/* mottsec(0,0) = boot device */
/* host = TFTP server */
/* h = host IP address */
/* e = target IP address */
/* u = username */
```

### 6. Basic Debugging

Workbench provides comprehensive debugging capabilities.

```c
/* Debug-friendly task example */
#include <vxWorks.h>
#include <taskLib.h>
#include <stdio.h>

int debugCounter = 0;

void debugTask(void)
{
    int localVar = 0;
    
    FOREVER
    {
        debugCounter++;
        localVar = debugCounter * 2;
        
        /* Set breakpoint here for debugging */
        printf("Counter: %d, Local: %d\n", debugCounter, localVar);
        
        taskDelay(60); /* 1 second delay */
    }
}
```

**Debugging Features:**
- **Breakpoints**: Set at source lines or addresses
- **Variable inspection**: View local and global variables
- **Call stack**: Examine function call hierarchy
- **Memory view**: Inspect memory contents
- **Task attachment**: Debug specific tasks

## Hands-On Exercises

### Exercise 1: Create Your First VxWorks Project

**Objective**: Set up a basic VxWorks project in Workbench

**Steps**:
1. Open Wind River Workbench
2. Create new VxWorks Image Project
3. Configure basic kernel parameters
4. Build the project
5. Connect to target (simulator or hardware)
6. Boot the VxWorks image

**Expected Output**: Successfully booted VxWorks system with shell prompt

### Exercise 2: Implement a Simple Task

**Objective**: Create and run a basic VxWorks task

**Task**: Implement the following task that prints a message every 2 seconds:

```c
/* Your implementation here */
void heartbeatTask(void)
{
    // TODO: Implement heartbeat task
    // Print "Heartbeat: <counter>" every 2 seconds
    // Use taskDelay() for timing
}

/* Spawn function */
int startHeartbeat(void)
{
    // TODO: Spawn the heartbeat task
    // Use priority 150, stack size 4096
    return OK;
}
```

**Expected Output**: 
```
Heartbeat: 1
Heartbeat: 2
Heartbeat: 3
...
```

### Exercise 3: Use Kernel Shell Commands

**Objective**: Practice essential kernel shell operations

**Tasks**:
1. Display all running tasks using `i` command
2. Show memory statistics with `memShow`
3. Load and run your heartbeat task
4. Suspend and resume the task
5. Check task stack usage

**Commands to practice**:
```bash
-> i
-> memShow
-> ld < heartbeat.out
-> startHeartbeat
-> ts "tHeartbeat"
-> tr "tHeartbeat"
-> checkStack "tHeartbeat"
```

### Exercise 4: Debug a Task

**Objective**: Use Workbench debugger to inspect task execution

**Steps**:
1. Set breakpoints in your heartbeat task
2. Attach debugger to the running task
3. Step through code execution
4. Inspect variable values
5. Examine call stack

**Debug Points**:
- Before taskDelay() call
- Inside the loop counter increment
- At printf() statement

### Exercise 5: Target Connection Setup

**Objective**: Configure and establish target connection

**Tasks**:
1. Create new target connection in Workbench
2. Configure connection parameters (IP, protocol)
3. Test connection to target
4. Upload and run application on target

**Connection Types to Try**:
- QEMU simulator
- Hardware target (if available)
- Wind River Simics (if available)

## Summary

In this module, you learned the fundamentals of VxWorks real-time operating system:

- **VxWorks Architecture**: Understanding the monolithic kernel design and key components
- **Workbench IDE**: Using the development environment for project management and debugging
- **Kernel Shell**: Essential commands for system interaction and task management
- **Real-Time Tasks**: Creating, spawning, and managing tasks in VxWorks
- **Target Configuration**: Setting up connections to hardware and simulators
- **Basic Debugging**: Using Workbench tools to debug applications

**Key Takeaways**:
- VxWorks provides deterministic real-time performance for embedded systems
- Tasks are the fundamental execution units with priority-based scheduling
- The kernel shell offers powerful command-line tools for development and debugging
- Workbench integrates all development tools in a unified environment
- Proper target configuration is essential for application deployment

**Next Steps**:
- Explore advanced VxWorks features like semaphores and message queues
- Learn about Real-Time Processes (RTPs) for memory-protected applications
- Study interrupt handling and device driver development
- Practice with more complex multi-task applications