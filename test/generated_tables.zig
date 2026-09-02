const std = @import("std");

const CheckError = error{
    InvalidArguments,
    FlashLutDrift,
    SwigluLutDrift,
    DuplicateSwigluLut,
};

pub fn main(init: std.process.Init) !void {
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer args.deinit();
    _ = args.next();

    const flash_generated_path = args.next() orelse return error.InvalidArguments;
    const flash_committed_path = args.next() orelse return error.InvalidArguments;
    const swiglu_generated_path = args.next() orelse return error.InvalidArguments;
    const swiglu_committed_path = args.next() orelse return error.InvalidArguments;
    if (args.next() != null) return error.InvalidArguments;

    const flash_generated = try readFile(init, flash_generated_path, 128 * 1024);
    defer init.gpa.free(flash_generated);
    const flash_committed = try readFile(init, flash_committed_path, 128 * 1024);
    defer init.gpa.free(flash_committed);
    if (!std.mem.eql(u8, flash_generated, flash_committed))
        return error.FlashLutDrift;

    const swiglu_generated = try readFile(init, swiglu_generated_path, 128 * 1024);
    defer init.gpa.free(swiglu_generated);
    const swiglu_committed = try readFile(init, swiglu_committed_path, 256 * 1024);
    defer init.gpa.free(swiglu_committed);
    const first = std.mem.indexOf(u8, swiglu_committed, swiglu_generated) orelse
        return error.SwigluLutDrift;
    if (std.mem.indexOfPos(u8, swiglu_committed, first + 1, swiglu_generated) != null)
        return error.DuplicateSwigluLut;
}

fn readFile(init: std.process.Init, path: []const u8, max_bytes: usize) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(init.io, path, init.gpa, .limited(max_bytes));
}
