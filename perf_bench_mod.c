#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/perf_event.h>
#include <linux/ktime.h>
#include <linux/slab.h>
#include <linux/sched.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Liam Strand");
MODULE_DESCRIPTION("Perf Event Enable/Disable Benchmark");

static unsigned long ids[] = { 0xc0, 145, 142, 138, 139 };
static struct perf_event *events[5];

static int __init perf_bench_init(void)
{
    struct perf_event_attr attr;
    int i;
    int iterations = 10000000;
    ktime_t start, end;
    s64 elapsed_ns, avg_ns;
    
    printk(KERN_INFO "Initializing module...\n");

    for (i = 0; i < 5; i++) {
        memset(&attr, 0, sizeof(struct perf_event_attr));
        attr.type = PERF_TYPE_RAW;
        attr.size = sizeof(struct perf_event_attr);
        attr.config = ids[i];
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;

        events[i] = perf_event_create_kernel_counter(&attr, -1, current, NULL, NULL);

        if (IS_ERR(events[i])) {
            printk(KERN_ERR "Failed to create event %d, err: %ld\n", i, PTR_ERR(events[i]));
            while (--i >= 0) {
                perf_event_release_kernel(events[i]);
            }
            return PTR_ERR(events[i]);
        }
    }

    for (i = 0; i < 4; i++) {
        perf_event_enable(events[i]);
    }

    printk(KERN_INFO "Benchmarking %d iterations of ENABLE/DISABLE pairs...\n", iterations);

    start = ktime_get();

    for (i = 4; i < iterations + 4; i++) {
        perf_event_disable(events[(i - 4) % 5]);
        perf_event_enable(events[i % 5]);
    }

    end = ktime_get();

    elapsed_ns = ktime_to_ns(ktime_sub(end, start));
    
    avg_ns = elapsed_ns;
    {
        u64 div = iterations * 2;
        do_div(avg_ns, div);
    }

    printk(KERN_INFO "Total time: %lld ns\n", elapsed_ns);
    {
        s64 sec = elapsed_ns;
        u64 rem;
        rem = do_div(sec, 1000000000);
        printk(KERN_INFO "Total time: %lld.%09llu s\n", sec, rem);
    }
    printk(KERN_INFO "Average time per operation: %lld ns\n", avg_ns);

    return 0;
}

static void __exit perf_bench_exit(void)
{
    int i;
    for (i = 0; i < 5; i++) {
        if (!IS_ERR_OR_NULL(events[i])) {
            perf_event_release_kernel(events[i]);
        }
    }
    printk(KERN_INFO "Module unloaded.\n");
}

module_init(perf_bench_init);
module_exit(perf_bench_exit);
