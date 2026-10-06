import { app } from "../../scripts/app.js";

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (!("__h3OriginalType" in widget)) {
        widget.__h3OriginalType = widget.type;
        widget.__h3OriginalComputeSize = widget.computeSize;
        widget.__h3OriginalHidden = Boolean(widget.hidden);
    }
    // ComfyUI frontend 1.51+ / LiteGraph Node 2.0 filters widgets by the
    // `hidden` property. Older canvases and DOM widgets still key off the
    // widget type and computeSize, so keep all three mechanisms in sync.
    widget.hidden = visible ? widget.__h3OriginalHidden : true;
    widget.type = visible ? widget.__h3OriginalType : "hidden";
    widget.computeSize = visible ? widget.__h3OriginalComputeSize : () => [0, -4];
}

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function refreshModelStack(node) {
    setWidgetVisible(widget(node, "turbo_strength"), widget(node, "turbo_lora")?.value !== "NONE");
    setWidgetVisible(widget(node, "secondary_strength"), widget(node, "secondary_lora")?.value !== "NONE");
    const preset = String(widget(node, "attention_backend")?.value ?? "");
    const attentionLabel = preset.split(" • ")[0] || "ATTENTION";
    node.title = `2 • H3 MODEL STACK • ${attentionLabel}`;
    node.setSize(node.computeSize());
    app.graph?.setDirtyCanvas(true, true);
}

function parseLeadingInteger(value, fallback) {
    const match = String(value ?? "").match(/^\s*(\d+)/);
    return match ? Number(match[1]) : fallback;
}

function refreshMaster(node) {
    const preset = widget(node, "resolution_preset")?.value;
    const manual = preset === "CUSTOM native size";
    setWidgetVisible(widget(node, "aspect_ratio"), !manual);
    setWidgetVisible(widget(node, "manual_width"), manual);
    setWidgetVisible(widget(node, "manual_height"), manual);

    const countWidget = widget(node, "section_count");
    const count = Math.max(1, Math.min(12, Number(countWidget?.value ?? 6)));
    const editWidget = widget(node, "edit_section");
    if (editWidget && Number(editWidget.value) > count) editWidget.value = count;
    const edit = Math.max(1, Math.min(count, Number(editWidget?.value ?? 1)));
    for (let index = 1; index <= 12; index += 1) {
        setWidgetVisible(widget(node, `prompt_s${index}`), index === edit);
    }

    const sectionFrames = Number(widget(node, "section_frames")?.value ?? 141);
    const contextFrames = parseLeadingInteger(widget(node, "context_profile")?.value, 22);
    const finalFrames = sectionFrames + (count - 1) * (sectionFrames - contextFrames);
    const seconds = Math.max(0, finalFrames / 24);
    const sectionLabel = count === 1 ? "section" : "sections";
    const pipeline = node.comfyClass === "H3LVREF2VAMaster" ? "REF MASTER" : "MASTER";
    node.title = `1 • ${pipeline} • ${count} ${sectionLabel} • ${finalFrames}F • ${seconds.toFixed(2)}s`;
    node.setSize(node.computeSize());
    app.graph?.setDirtyCanvas(true, true);
}

function wrapCallback(target, callback) {
    if (!target || target.__h3Wrapped) return;
    target.__h3Wrapped = true;
    const original = target.callback;
    target.callback = function (...args) {
        const result = original?.apply(this, args);
        callback();
        return result;
    };
}

app.registerExtension({
    name: "h3_direct_latent_head.native_controls",
    nodeCreated(node) {
        if (node.comfyClass === "H3LVModelStack") {
            wrapCallback(widget(node, "turbo_lora"), () => refreshModelStack(node));
            wrapCallback(widget(node, "secondary_lora"), () => refreshModelStack(node));
            wrapCallback(widget(node, "attention_backend"), () => refreshModelStack(node));
            setTimeout(() => refreshModelStack(node), 0);
        }
        if (node.comfyClass === "H3LVFL2VAMaster" || node.comfyClass === "H3LVREF2VAMaster") {
            for (const name of [
                "resolution_preset",
                "section_count",
                "edit_section",
                "section_frames",
                "context_profile",
            ]) {
                wrapCallback(widget(node, name), () => refreshMaster(node));
            }
            setTimeout(() => refreshMaster(node), 0);
        }
    },
});
